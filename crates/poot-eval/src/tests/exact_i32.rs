use poot_tensor::DType;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use poot_graph_ir::{BinOp, Builder, Scalar, StateRole, TensorType, UnOp};
use poot_load::packed_safetensors::{
    AuthenticatedInventory, AuthenticatedSafetensorsHandleSet, ExactSourceOwner,
    ExactSourceOwnerCache, InventoryDecision, PackedArtifactManifest, PackedOwnerCache,
    PackedSafetensorsLimits, PackedShardManifest, TensorDisposition, sha256_digest,
};

use super::allocation::allocated_bytes;
use super::helpers;
use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_tensor::HostTensor;

static NEXT_EXACT_I32_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct ExactI32TempFixture {
    path: PathBuf,
}

impl ExactI32TempFixture {
    fn new() -> Self {
        let sequence = NEXT_EXACT_I32_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("poot-exact-i32-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for ExactI32TempFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(super) fn authenticated_source_owner(
    name: &str,
    dtype: &str,
    elements: usize,
    payload: &[u8],
    disposition: TensorDisposition,
) -> Arc<ExactSourceOwner> {
    let fixture = ExactI32TempFixture::new();
    let config = b"{}";
    fs::write(fixture.path.join("config.json"), config).unwrap();
    let byte_len = payload.len();
    let header = format!(
        "{{\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{elements}],\"data_offsets\":[0,{byte_len}]}}}}"
    );
    let mut shard = Vec::new();
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(header.as_bytes());
    shard.extend_from_slice(payload);
    let shard_name = "model.safetensors";
    fs::write(fixture.path.join(shard_name), &shard).unwrap();
    let index = format!(r#"{{"weight_map":{{"{name}":"model.safetensors"}}}}"#);
    let index = index.as_bytes();
    fs::write(fixture.path.join("model.safetensors.index.json"), index).unwrap();
    let manifest = PackedArtifactManifest {
        repository: "local/exact-i32-fixture".to_string(),
        revision: "card371".to_string(),
        config_length: config.len(),
        config_sha256: sha256_digest(config),
        index_sha256: sha256_digest(index),
        shards: vec![PackedShardManifest {
            filename: shard_name.to_string(),
            file_length: shard.len(),
            file_sha256: sha256_digest(&shard),
            header_length: header.len(),
            header_sha256: sha256_digest(header.as_bytes()),
        }],
    };
    let limits = PackedSafetensorsLimits {
        config_bytes: 1024,
        index_bytes: 1024,
        header_bytes_per_shard: 1024,
        shard_count: 1,
        tensor_entries: 1,
        selected_source_bytes: byte_len,
        packed_source_bytes: 1,
    };
    let mut authenticated =
        AuthenticatedSafetensorsHandleSet::authenticate(&fixture.path, manifest, limits).unwrap();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let result = authenticated
        .load_mixed(
            &mut packed_cache,
            &mut exact_cache,
            |inventory: AuthenticatedInventory<'_>| {
                Ok::<_, &'static str>(
                    inventory
                        .rows()
                        .map(|row| InventoryDecision::new(row.key(), disposition.clone()))
                        .collect::<Vec<_>>(),
                )
            },
        )
        .unwrap();
    Arc::clone(&result.exact_metadata.values().next().unwrap().owner)
}

fn i32_tensor(shape: Vec<usize>, values: &[i32]) -> HostTensor {
    HostTensor::i32(shape, values.to_vec())
}

#[test]
fn exact_i32_binary_uses_authoritative_words_with_scalar_and_broadcast_forms() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2, 1], DType::I32));
    let y = b.constant("y", TensorType::new(vec![1, 3], DType::I32));
    let sum = b.binary(BinOp::Add, x, y);
    let shifted = b.binary_scalar(BinOp::Sub, sum, Scalar::I32(1));
    let out = b.binary_scalar(BinOp::Mul, shifted, Scalar::I32(-1));
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        x.id,
        Value::from(i32_tensor(vec![2, 1], &[16_777_217, i32::MAX])),
    );
    inputs.insert(y.id, Value::from(i32_tensor(vec![1, 3], &[0, 1, i32::MIN])));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let expected = [
        -16_777_216,
        -16_777_217,
        2_130_706_432,
        -2_147_483_646,
        -2_147_483_647,
        2,
    ];
    assert_eq!(got.shape(), vec![2, 3]);
    assert_eq!(got.as_i32(), Some(expected.as_slice()));

    // 2^24+1 is not an f32: the bound input keeps its exact word and the computed output is an
    // I32 tensor with no f32 payload, so the exact result cannot have gone through an f32 rounding.
    let Value::Host(x_tensor) = &inputs[&x.id] else {
        panic!("expected a host I32 carrier for x")
    };
    assert_eq!(x_tensor.as_i32().unwrap()[0], 16_777_217);
    assert_eq!(got.dtype(), DType::I32);
    assert!(got.as_f32().is_none());
}

#[test]
fn signed_and_unsigned_i32_comparisons_diverge_at_the_sign_bit() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![4], DType::I32));
    let zero = b.constant("zero", TensorType::new(vec![4], DType::I32));
    let max = b.binary(BinOp::Max, x, zero);
    let signed = b.binary(BinOp::Ge, x, zero);
    let unsigned = b.binary(BinOp::GeU, x, zero);
    let g = b.finish(unsigned);
    let mut inputs = HashMap::new();
    inputs.insert(
        x.id,
        Value::from(i32_tensor(vec![4], &[0, 1, i32::MAX, i32::MIN])),
    );
    inputs.insert(zero.id, Value::from(i32_tensor(vec![4], &[0; 4])));
    let env = eval(
        &g,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap();
    assert_eq!(
        env[max.id].as_ref().unwrap().as_host().unwrap().as_i32(),
        Some(&[0, 1, i32::MAX, 0][..])
    );
    assert_eq!(
        env[signed.id].as_ref().unwrap().as_host().unwrap().as_i32(),
        Some(&[1, 1, 1, 0][..])
    );
    assert_eq!(
        env[unsigned.id]
            .as_ref()
            .unwrap()
            .as_host()
            .unwrap()
            .as_i32(),
        Some(&[1, 1, 1, 1][..])
    );
}

/// Literal expected values for `RemU`, including operands at and above 2^31 (the u32 interpretation
/// boundary) where signed `%` would disagree. The signed-rem mutation of the oracle turns the
/// second and third lanes red.
#[test]
fn unsigned_remainder_matches_u32_modulus_literals_above_the_sign_bit() {
    // dividend bit patterns as u32: 7, 0x8000_0000, 0xffff_ffff, 0xffff_fffe, 0
    let dividends = [7, i32::MIN, -1, -2, 0];
    let divisors = [5, 3, 7, 0x7fff_ffffu32 as i32, 9];
    // u32 expectations: 7%5=2, 2^31%3=2, (2^32-1)%7=0, (2^32-2)%0x7fffffff=0x7ffffffe, 0%9=0.
    let good = [
        2,
        (1u32 << 31).wrapping_rem(3) as i32,
        u32::MAX.wrapping_rem(7) as i32,
        (u32::MAX - 1).wrapping_rem(0x7fff_ffff) as i32,
        0,
    ];
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![5], DType::I32));
    let m = b.constant("m", TensorType::new(vec![5], DType::I32));
    let out = b.binary(BinOp::RemU, x, m);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(x.id, Value::from(i32_tensor(vec![5], &dividends)));
    inputs.insert(m.id, Value::from(i32_tensor(vec![5], &divisors)));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(got.as_i32(), Some(&good[..]), "u32 remainder literals");
    // Signed % would give different answers for the high lanes: document the divergence.
    let signed = |a: i32, b: i32| a.wrapping_rem(b);
    assert_ne!(
        good[1],
        signed(dividends[1], divisors[1]),
        "lane 1 must distinguish unsigned from signed remainder"
    );
    assert_ne!(
        good[2],
        signed(dividends[2], divisors[2]),
        "lane 2 must distinguish unsigned from signed remainder"
    );

    // Zero divisor fails closed.
    let b0 = Builder::new();
    let x0 = b0.constant("x", TensorType::new(vec![1], DType::I32));
    let m0 = b0.constant("m", TensorType::new(vec![1], DType::I32));
    let out0 = b0.binary(BinOp::RemU, x0, m0);
    let g0 = b0.finish(out0);
    let err = eval(
        &g0,
        &HashMap::from([
            (x0.id, Value::from(i32_tensor(vec![1], &[1]))),
            (m0.id, Value::from(i32_tensor(vec![1], &[0]))),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("RemU") && message.contains("divide by zero"),
        "zero divisor must fail closed with RemU in the message: {message}"
    );
}

#[test]
fn i32_broadcast_preserves_words_and_bounded_index_inputs_remain_compatible() {
    let b = Builder::new();
    let words = b.constant("words", TensorType::new(vec![2, 1], DType::I32));
    let out = b.broadcast(words, vec![2, 3]);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        words.id,
        Value::from(i32_tensor(vec![2, 1], &[16_777_217, i32::MIN])),
    );
    assert_eq!(
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_i32(),
        Some(
            &[
                16_777_217,
                16_777_217,
                16_777_217,
                i32::MIN,
                i32::MIN,
                i32::MIN
            ][..]
        )
    );

    // The unified walk's bind-time preflight requires every I32 input to
    // carry authoritative `the I32 words`, even an index operand that only ever feeds a non-arithmetic
    // Gather; the old lossy-f32-mirror bind this row used to exercise no longer admits at all (there is
    // no range check to pass or fail - the bind itself is refused). Bound authoritatively here, this
    // keeps checking the thing the name still promises: a Gather index does not need the exact-I32 word
    // lane, a plain dense I32 bind is enough.
    let b = Builder::new();
    let table = b.constant("table", TensorType::f32(vec![3, 2]));
    let index = b.constant("index", TensorType::scalar(DType::I32));
    let row = b.gather(table, 0, index);
    let g = b.finish(row);
    let mut inputs = HashMap::new();
    inputs.insert(
        table.id,
        Value::from(HostTensor::f32(
            vec![3, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )),
    );
    inputs.insert(index.id, Value::from(HostTensor::i32(vec![], vec![2])));
    assert_eq!(
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_f32()
            .unwrap(),
        &[5.0, 6.0]
    );
}

#[test]
fn i32_words_are_exact_and_a_non_i32_bind_is_refused() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![1], DType::I32));
    let out = b.binary_scalar(BinOp::Add, x, Scalar::I32(1));
    let g = b.finish(out);
    // 2^24 + 1 is not an f32: the sum is exact only because the words are never widened.
    let mut inputs = HashMap::from([(
        x.id,
        Value::from(HostTensor::i32(vec![1], vec![16_777_217])),
    )]);
    assert_eq!(
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_i32(),
        Some(&[16_777_218][..])
    );

    // An F32 tensor is not an I32 input: there is no implicit conversion at bind.
    inputs.insert(
        x.id,
        Value::from(HostTensor::f32(vec![1], vec![16_777_216.0])),
    );
    let error = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap_err()
        .to_string();
    assert!(error.contains("different dtype"), "{error}");
}

#[test]
fn exact_concat_combines_two_authoritative_i32_parts() {
    // Concat of two I32 parts does not lose words (the unified walk requires I32 inputs to be I32
    // tensors, so there is no "legacy f32-backed" part to promote).
    let b = Builder::new();
    let exact = b.constant("exact", TensorType::new(vec![1], DType::I32));
    let legacy = b.constant("legacy", TensorType::new(vec![1], DType::I32));
    let joined = b.concat(0, &[exact, legacy]);
    let out = b.binary_scalar(BinOp::Add, joined, Scalar::I32(0));
    let g = b.finish(out);
    let inputs = HashMap::from([
        (
            exact.id,
            Value::from(HostTensor::i32(vec![1], vec![i32::MIN])),
        ),
        (legacy.id, Value::from(HostTensor::i32(vec![1], vec![7]))),
    ]);
    let env = eval(
        &g,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap();
    assert_eq!(
        env[joined.id].as_ref().unwrap().as_host().unwrap().as_i32(),
        Some(&[i32::MIN, 7][..])
    );
    assert_eq!(
        env[out.id].as_ref().unwrap().as_host().unwrap().as_i32(),
        Some(&[i32::MIN, 7][..])
    );
}

#[test]
fn exact_i32_scalar_empty_overflow_and_error_precedence_are_total() {
    let b = Builder::new();
    let scalar = b.constant("scalar", TensorType::scalar(DType::I32));
    let wrapped = b.binary_scalar(BinOp::Add, scalar, Scalar::I32(1));
    let scalar_graph = b.finish(wrapped);
    assert_eq!(
        eval(
            &scalar_graph,
            &HashMap::from([(
                scalar.id,
                Value::from(HostTensor::i32(vec![], vec![i32::MAX]))
            )]),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .unwrap()
        .as_i32(),
        Some(&[i32::MIN][..])
    );

    let b = Builder::new();
    let empty = b.constant("empty", TensorType::new(vec![0], DType::I32));
    let out = b.binary_scalar(BinOp::Mul, empty, Scalar::I32(3));
    let mut empty_graph = b.finish(out);
    let empty_words = HostTensor::i32(vec![0], vec![]);
    empty_graph.eqns[0].op = poot_graph_ir::OpKind::Binary(BinOp::Div);
    let error = eval(
        &empty_graph,
        &HashMap::from([(empty.id, Value::from(empty_words))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("Binary(Div)"), "{error}");

    let _ = (empty_graph, out);

    // Card 554d: `ExactI32Error::ElementCountOverflow` is gone with the whole exact-I32 lane. The
    // dense walk's I32 `Binary`/`Select` shape overflow now reaches the one shared
    // `ops::element_count` helper (`ops/mod.rs`), which reports the typed, eqn-free
    // `EvalError::ElementCountOverflow` instead of a mislabeled
    // `Unsupported` with a bogus equation id. Card 554d's own `g.validate()` (run before any
    // equation evaluates) now also catches a declared-aval/inferred-aval mismatch earlier than this
    // overflow guard, so a crafted malformed *graph* can no longer reach `element_count` through
    // `eval` - the guard is exercised directly instead, the way `ops/mod.rs`'s own doc already frames
    // it as one lane-neutral helper, not something only a full graph eval can prove.
    let error = crate::ops::element_count(&[usize::MAX, 2]).unwrap_err();
    let crate::EvalError::ElementCountOverflow { shape } = &error else {
        panic!("expected ElementCountOverflow, got {error:?}")
    };
    assert_eq!(*shape, vec![usize::MAX, 2]);
}

#[test]
fn exact_i32_bit_select_and_clz_match_wrapping_word_semantics() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![4], DType::I32));
    let y = b.constant("y", TensorType::new(vec![4], DType::I32));
    let and = b.binary(BinOp::And, x, y);
    let or = b.binary(BinOp::Or, x, y);
    let xor = b.binary(BinOp::Xor, x, y);
    let shl = b.binary_scalar(BinOp::Shl, x, Scalar::I32(33));
    let shr = b.binary(BinOp::Shr, x, y);
    let not = b.unary(UnOp::Not, x);
    let clz = b.unary(UnOp::Clz, x);
    let selected = b.select(and, or, xor);
    let g = b.finish(selected);
    let mut inputs = HashMap::new();
    inputs.insert(
        x.id,
        Value::from(i32_tensor(vec![4], &[0, -1, i32::MIN, 1])),
    );
    inputs.insert(y.id, Value::from(i32_tensor(vec![4], &[7, 0, 1, 33])));
    let env = eval(
        &g,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .unwrap()
    .environment
    .unwrap();
    let dense_ints = |id: poot_graph_ir::ValueId| -> Option<Vec<i32>> {
        env[id]
            .as_ref()
            .unwrap()
            .as_host()
            .unwrap()
            .as_i32()
            .map(<[i32]>::to_vec)
    };
    assert_eq!(dense_ints(and.id), Some(vec![0, 0, 0, 1]));
    assert_eq!(dense_ints(or.id), Some(vec![7, -1, i32::MIN | 1, 33]));
    assert_eq!(dense_ints(xor.id), Some(vec![7, -1, i32::MIN ^ 1, 32]));
    assert_eq!(dense_ints(shl.id), Some(vec![0, -2, 0, 2]));
    assert_eq!(
        dense_ints(shr.id),
        Some(vec![0, -1, ((i32::MIN as u32) >> 1) as i32, 0])
    );
    assert_eq!(dense_ints(not.id), Some(vec![-1, 0, i32::MAX, -2]));
    assert_eq!(dense_ints(clz.id), Some(vec![32, 0, 0, 31]));
    assert_eq!(dense_ints(selected.id), Some(vec![7, -1, i32::MIN ^ 1, 33]));
}

#[test]
fn clz_of_zero_and_sign_bit_are_defined() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2], DType::I32));
    let out = b.unary(UnOp::Clz, x);
    let g = b.finish(out);
    let got = eval(
        &g,
        &HashMap::from([(x.id, Value::from(i32_tensor(vec![2], &[0, i32::MIN])))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert_eq!(got.as_i32(), Some(&[32, 0][..]));
}

#[test]
fn select_wrapping_formula_is_not_a_boolean_select() {
    let b = Builder::new();
    let cond = b.constant("cond", TensorType::new(vec![2], DType::I32));
    let if_true = b.constant("t", TensorType::new(vec![2], DType::I32));
    let if_false = b.constant("f", TensorType::new(vec![2], DType::I32));
    let out = b.select(cond, if_true, if_false);
    let g = b.finish(out);
    let got = eval(
        &g,
        &HashMap::from([
            (cond.id, Value::from(i32_tensor(vec![2], &[2, -1]))),
            (if_true.id, Value::from(i32_tensor(vec![2], &[10, 10]))),
            (if_false.id, Value::from(i32_tensor(vec![2], &[3, 3]))),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert_eq!(got.as_i32(), Some(&[17, -4][..]));
}

#[test]
fn exact_i32_owner_identity_is_not_word_equality() {
    use crate::ExactI32TensorView;

    let first_owner: Arc<[i32]> = Arc::from([7, 11]);
    let equal_foreign_owner: Arc<[i32]> = Arc::from([7, 11]);
    let first = ExactI32TensorView::try_from_words(vec![2], Arc::clone(&first_owner)).unwrap();
    let shared = ExactI32TensorView::try_from_words(vec![2], Arc::clone(&first_owner)).unwrap();
    let equal_foreign =
        ExactI32TensorView::try_from_words(vec![2], Arc::clone(&equal_foreign_owner)).unwrap();

    assert_eq!(first.i32_words(), equal_foreign.i32_words());
    assert_ne!(first_owner.as_ptr(), equal_foreign_owner.as_ptr());
    assert_eq!(first, shared);
    assert_ne!(first, equal_foreign);
}

fn exact_view<const N: usize>(shape: Vec<usize>, words: [i32; N]) -> crate::Value {
    crate::Value::Owner(crate::ExactValue::I32(
        crate::ExactI32TensorView::try_from_words(shape, Arc::from(words)).unwrap(),
    ))
}

/// Card 396: an owner-backed BF16 dense view over `values`, encoded as the top half of the f32 bits
/// (as `exact_dense::decode_bf16_word` decodes). `authenticated_source_owner` runs full authentication.
fn owner_backed_bf16_dense(name: &str, values: &[f32]) -> crate::exact_dense::DenseOwnerTensorView {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let owner = authenticated_source_owner(
        name,
        "BF16",
        values.len(),
        &bytes,
        TensorDisposition::DenseBf16,
    );
    crate::exact_dense::DenseOwnerTensorView::new(owner).unwrap()
}

/// The F32 counterpart of [`owner_backed_bf16_dense`].
fn owner_backed_f32_dense(name: &str, values: &[f32]) -> crate::exact_dense::DenseOwnerTensorView {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let owner = authenticated_source_owner(
        name,
        "F32",
        values.len(),
        &bytes,
        TensorDisposition::DenseF32,
    );
    crate::exact_dense::DenseOwnerTensorView::new(owner).unwrap()
}

/// One `Gather(data, 0, index)` graph, `data` a 1-D constant of `rows` elements at `dtype`.
fn gather_graph(
    rows: usize,
    dtype: DType,
    index_count: usize,
) -> (
    Builder,
    poot_graph_ir::Traced,
    poot_graph_ir::Traced,
    poot_graph_ir::Traced,
) {
    let builder = Builder::new();
    let table = builder.constant("table", TensorType::new(vec![rows], dtype));
    let index = helpers::i32_constant(&builder, "index", vec![index_count]).unwrap();
    let output = builder.gather(table, 0, index);
    (builder, table, index, output)
}

/// Card 396: a Gather over an owner-backed BF16 data source reads the right elements.
///
/// Red mutation: remove the `Value::Owner(exact @ (ExactValue::Dense(_) | ExactValue::Bf16(_)))` arm in
/// the `Gather` else-branch of `exact_i32.rs`; the source then hits
/// `UnsupportedOperation { operation: "Gather with non-I32 exact data" }`.
#[test]
fn exact_i32_gather_reads_an_owner_backed_bf16_data_source() {
    use crate::ExactValue;

    let (builder, table, index, output) = gather_graph(4, DType::BF16, 2);
    let graph = builder.finish(output);
    let view = owner_backed_bf16_dense("table", &[1.5, -2.25, 0.0, 8.0]);
    let result = eval(
        &graph,
        &HashMap::from([
            (table.id, Value::Owner(ExactValue::Dense(view))),
            (index.id, exact_view(vec![2], [3, 1])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    // Card 554d: a Gather of a BF16 table by I32 index is one answer now, not two - the
    // same Spec 376 admission the Exact-BF16 lane always used, keeping the result BF16-exact rather
    // than widening it the moment any I32 index touches it.
    let Value::Owner(ExactValue::Bf16(result)) = result else {
        panic!("expected an exact BF16 gather result, got {result:?}")
    };
    assert_eq!(
        (0..result.numel())
            .map(|i| result.value(i))
            .collect::<Vec<_>>(),
        vec![8.0, -2.25]
    );
}

/// Card 396: the same over the F32 dense carrier.
///
/// Red mutation: as above.
#[test]
fn exact_i32_gather_reads_an_owner_backed_f32_data_source() {
    use crate::ExactValue;

    let (builder, table, index, output) = gather_graph(4, DType::F32, 2);
    let graph = builder.finish(output);
    let view = owner_backed_f32_dense("table", &[1.5, -2.25, 0.125, 8.0]);
    let result = eval(
        &graph,
        &HashMap::from([
            (table.id, Value::Owner(ExactValue::Dense(view))),
            (index.id, exact_view(vec![2], [2, 0])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("expected a dense gather result, got {result:?}")
    };
    assert_eq!(result.as_f32().unwrap(), &[0.125, 1.5]);
}

/// Card 396: index bounds are still checked on the owner-backed path, against the owner view's retained
/// shape (the graph's `aval` shape is kept in agreement so only the bounds check is exercised).
///
/// Red mutation: delete the `index_at` call in the Spec-376 Gather arm (`walk::evaluate_bf16`), so an
/// out-of-range index reads past the owner's extent.
#[test]
fn exact_i32_gather_still_checks_bounds_on_an_owner_backed_data_source() {
    use crate::ops::index_rule::IndexFaultKind;
    use crate::{EvalError, ExactValue, Value};

    let (builder, table, index, output) = gather_graph(2, DType::BF16, 1);
    let graph = builder.finish(output);
    let view = owner_backed_bf16_dense("table", &[1.0, 2.0]);
    let result = eval(
        &graph,
        &HashMap::from([
            (table.id, Value::Owner(ExactValue::Dense(view))),
            (index.id, exact_view(vec![1], [5])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    );
    // Card 555: a BF16 table + I32 index Gather is one answer, the same Spec 376
    // admission the Exact-BF16 lane always used, so the bounds check is now the one index rule's
    // `EvalError::Index`, not `ExactBf16Error::GatherIndex` or a plain `gather` `Unsupported` refusal.
    let Err(EvalError::Index(fault)) = &result else {
        panic!("expected a bounds rejection, got {result:?}")
    };
    assert_eq!(fault.position, 0);
    assert_eq!(fault.len, 2);
    assert_eq!(fault.kind, IndexFaultKind::OutOfRange);
}

/// Card 396: the owner-backed path and the plain-`Value::Host` path agree on the same bytes.
///
/// Red mutation: make `gather_exact_dense` ignore `axis`; this graph's `axis` is 1, so an axis-blind read
/// diverges from the plain-dense oracle over the identical bytes.
#[test]
fn exact_i32_gather_owner_backed_path_agrees_with_the_plain_dense_path() {
    use crate::{ExactValue, Value};

    let builder = Builder::new();
    let table = builder.constant("table", TensorType::new(vec![2, 3], DType::BF16));
    let index = helpers::i32_constant(&builder, "index", vec![2]).unwrap();
    let output = builder.gather(table, 1, index);
    let graph = builder.finish(output);

    // Every value is a power of two (or zero), so the BF16 round trip is exact: this checks agreement
    // between the two read paths, not BF16 rounding.
    let values = [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0];
    // `authenticated_source_owner` declares a flat `[elements]` shape; reshape to the graph's `[2, 3]`.
    let view = owner_backed_bf16_dense("table", &values)
        .reshape(vec![2, 3])
        .unwrap();
    let owner_inputs = HashMap::from([
        (table.id, Value::Owner(ExactValue::Dense(view))),
        (index.id, exact_view(vec![2], [2, 0])),
    ]);
    let dense_inputs = HashMap::from([
        (
            table.id,
            Value::Host(crate::ops::cast::narrow_f32(vec![2, 3], &values, DType::BF16).unwrap()),
        ),
        (index.id, exact_view(vec![2], [2, 0])),
    ]);

    let owner_result = eval(
        &graph,
        &owner_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let dense_result = eval(
        &graph,
        &dense_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;

    // Card 554d: both bindings are a BF16 table by I32 index, so both now take the one
    // Spec 376 Gather path and publish the same exact-BF16 carrier, not a dense result.
    let (
        Value::Owner(ExactValue::Bf16(owner_result)),
        Value::Owner(ExactValue::Bf16(dense_result)),
    ) = (owner_result, dense_result)
    else {
        panic!("expected both paths to produce an exact BF16 gather result")
    };
    let words = |v: &crate::exact_bf16::ExactBf16TensorView| {
        (0..v.numel()).map(|i| v.value(i)).collect::<Vec<_>>()
    };
    assert_eq!(words(&owner_result), words(&dense_result));
    assert_eq!(words(&owner_result), vec![4.0, 1.0, 32.0, 8.0]);
}

/// Card 396: gathering a few rows out of a large owner-backed source does not materialize the whole
/// source, the falsifiable form of the memory-safety claim behind `gather_exact_dense` (Qwen3.5's
/// `embed.weight` is 248,320 x 5,120, ~5GB as F32).
///
/// Red mutation: replace `gather_exact_dense`'s lazy per-element read with
/// `ExactValue::materialize_dense`; materializing `ROWS` BF16 words into f32 allocates ~`ROWS * 4`
/// bytes regardless of how many are selected, failing this row's allocation bound.
#[test]
fn exact_i32_gather_does_not_materialize_the_whole_owner_backed_source() {
    use crate::{ExactValue, Value};

    const ROWS: usize = 100_000;
    let mut values = vec![0.0f32; ROWS];
    values[3] = 8.0;
    values[ROWS - 1] = -16.0;
    let view = owner_backed_bf16_dense("table", &values);

    let (builder, table, index, output) = gather_graph(ROWS, DType::BF16, 2);
    let graph = builder.finish(output);
    let inputs = HashMap::from([
        (table.id, Value::Owner(ExactValue::Dense(view))),
        (index.id, exact_view(vec![2], [3, ROWS as i32 - 1])),
    ]);

    let (result, bytes) = allocated_bytes(|| {
        eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|e| e.output)
    });
    // Card 554d: a BF16 table + I32 index Gather is one answer now (the Spec 376 lane),
    // which keeps the two selected rows BF16-exact rather than widening them to a dense F32 result.
    let Value::Owner(ExactValue::Bf16(result)) = result.unwrap() else {
        panic!("expected an exact BF16 gather result")
    };
    assert_eq!(
        (0..result.numel())
            .map(|i| result.value(i))
            .collect::<Vec<_>>(),
        vec![8.0, -16.0]
    );
    assert!(
        bytes < 10_000,
        "gathering 2 rows out of {ROWS} allocated {bytes} bytes on this thread - an eager f32 decode of \
         the whole source would need at least {} bytes, so this looks like the whole source was \
         materialized instead of read lazily",
        ROWS * 4,
    );
}

/// Card 554d: the deleted exact-I32 lane's own `dense_equation` blanket fallback -
/// silently materializing ANY owner-backed BF16 constant to F32 the moment some unrelated op touched
/// it - is gone with no replacement. The one walk's admission rule is strict and uniform instead
/// (`exact_dense::validate_exact_dense_bindings`, card 396's own doc): an owner view may feed only a
/// Spec 376 BF16 equation (`resolve::classify`) or the graph's own output; a plain `Binary` is
/// neither, so it is a refused `GraphConsumer`, not a convenience widen.
///
/// Red mutation: drop the `resolve::classify(g, eqn).is_some()` admission check in
/// `validate_exact_dense_bindings`; the owner-backed constant below is then read (not refused), and
/// this test's `unwrap_err` goes red.
#[test]
fn owner_backed_bf16_constant_consumed_by_a_plain_binary_is_refused() {
    use crate::exact_dense::ExactDenseError;
    use crate::{ExactValue, Value};

    let builder = Builder::new();
    let table = builder.constant("norm", TensorType::new(vec![2], DType::BF16));
    let output = builder.binary_scalar(BinOp::Add, table, Scalar::F32(1.0));
    let graph = builder.finish(output);

    let view = owner_backed_bf16_dense("norm", &[3.0, -1.0]);
    let error = eval(
        &graph,
        &HashMap::from([(table.id, Value::Owner(ExactValue::Dense(view)))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_err();
    let crate::EvalError::ExactDense(ExactDenseError::GraphConsumer {
        value_id, consumer, ..
    }) = &error
    else {
        panic!("expected a refused GraphConsumer, got {error:?}")
    };
    assert_eq!(*value_id, table.id);
    assert_eq!(consumer, "add");
}

#[test]
fn exact_i32_compare_sub_select_are_wordwise() {
    use crate::Value;

    let builder = Builder::new();
    let left = helpers::i32_constant(&builder, "left", vec![2]).unwrap();
    let right = helpers::i32_constant(&builder, "right", vec![2]).unwrap();
    let signed = builder.binary(BinOp::Ge, left, right);
    let unsigned = builder.binary(BinOp::GeU, left, right);
    let difference = builder.binary_scalar(BinOp::Sub, left, Scalar::I32(-1));
    let output = builder.select(difference, signed, unsigned);
    let graph = builder.finish(output);
    let inputs = HashMap::from([
        (left.id, exact_view(vec![2], [i32::MIN, i32::MAX])),
        (right.id, exact_view(vec![2], [0, -1])),
    ]);
    let result = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output;
    // Card 554d: the unified walk always publishes an I32 equation's output as `Value::Host`
    // (`HostTensor::i32`), never `Value::Owner(ExactValue::I32(..))`; the one carrier-preserving
    // exception is a plain `Reshape` of an exact-I32-bound input, which this graph's `Select` output
    // is not. The deleted `ExactI32Observer` accounting assertions this test used to carry
    // (`operation_derived_i32_allocations`/`ExactI32AllocationKind`/`retained_full_f32_mirror_bytes`)
    // have no replacement (flagged in the migration report); only the value-correctness half survives.
    let Value::Host(result) = result else {
        panic!("expected a dense I32 result, got {result:?}")
    };
    assert_eq!(result.as_i32(), Some(&[i32::MIN, i32::MIN][..]));

    let builder = Builder::new();
    let left = helpers::i32_constant(&builder, "tie.left", vec![1]).unwrap();
    let right = helpers::i32_constant(&builder, "tie.right", vec![1]).unwrap();
    let signed_tie = builder.binary(BinOp::Ge, left, right);
    let graph = builder.finish(signed_tie);
    let result = eval(
        &graph,
        &HashMap::from([
            (left.id, exact_view(vec![1], [7])),
            (right.id, exact_view(vec![1], [7])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("expected a dense I32 tie result, got {result:?}")
    };
    assert_eq!(result.as_i32(), Some(&[1][..]));
}

#[test]
fn exact_i32_gather_preserves_large_words() {
    use crate::{EvalError, Value};

    let builder = Builder::new();
    let table = helpers::i32_constant(&builder, "table", vec![3]).unwrap();
    let index = helpers::i32_constant(&builder, "index", vec![2]).unwrap();
    let output = builder.gather(table, 0, index);
    let graph = builder.finish(output);
    let table_value = exact_view(vec![3], [16_777_217, -7, i32::MAX]);
    let result = eval(
        &graph,
        &HashMap::from([
            (table.id, table_value.clone()),
            (index.id, exact_view(vec![2], [0, 2])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    // Card 554d: Gather over an exact-I32 table always publishes `Value::Host(HostTensor::i32(..))` now
    // (the unified walk's `evaluate_gather` builds a fresh word `Arc` for every I32-data Gather,
    // whichever carrier the table was bound with); `ExactI32AllocationKind`/`*_total_bytes` accounting
    // has no replacement (flagged in the migration report).
    let Value::Host(result) = result else {
        panic!("expected a dense Gather result, got {result:?}")
    };
    assert_eq!(result.as_i32(), Some(&[16_777_217, i32::MAX][..]));

    use crate::ops::index_rule::IndexFaultKind;
    for (bad, expected_kind) in [
        (-1, IndexFaultKind::Negative),
        (3, IndexFaultKind::OutOfRange),
    ] {
        let outcome = eval(
            &graph,
            &HashMap::from([
                (table.id, table_value.clone()),
                (index.id, exact_view(vec![2], [1, bad])),
            ]),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        );
        let Err(EvalError::Index(fault)) = &outcome else {
            panic!("expected a bounds rejection, got {outcome:?}")
        };
        assert_eq!(fault.position, 1);
        assert_eq!(fault.len, 3);
        assert_eq!(fault.kind, expected_kind);
    }
}

#[test]
fn selected_cast_checks_before_allocation() {
    // Card 554d: the deleted `ExactI32Observer`'s reservation-forcing seam
    // (`force_selected_cast_reservation_failure`, `ExactI32Error::SelectedCastReserve`,
    // `try_new_uninit_arc_slice`, and every `selected_cast_*`/`*_unpublished_peak_bytes` accounting
    // field) has no replacement: the unified walk's `Cast(I32 -> F32)` arm (`walk.rs::evaluate_cast`)
    // allocates through a plain `initialized_arc_slice` call with no fallible-reservation
    // instrumentation and no injectable failure point, and `EvalOptions`'s budget only gates a
    // caller-stated element/byte ceiling, not an injected allocator fault. That whole "a forged huge
    // selected-cast size fails cleanly before corrupting state" row is dropped here, flagged in the
    // migration report rather than reinvented in `walk.rs` (which this task may not edit). The three
    // still-live admission checks survive, migrated onto their new error shapes.
    use crate::cast_authority::CastAuthorityError;
    use crate::cast_authority::{ExactI32CastBounds, ExactI32CastRole};
    use crate::{EvalError, Value};

    let builder = Builder::new();
    let table = helpers::i32_constant(&builder, "table", vec![3]).unwrap();
    let index = helpers::i32_constant(&builder, "index", vec![2]).unwrap();
    let selected = builder.gather(table, 0, index);
    let cast = builder.cast(selected, DType::F32);
    let graph = builder.finish(cast);
    let valid_inputs = HashMap::from([
        (table.id, exact_view(vec![3], [0, 1, 2])),
        (index.id, exact_view(vec![2], [0, 2])),
    ]);
    let mut ceiling_role = |value: poot_graph_ir::ValueId| {
        (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(ExactI32CastBounds {
            selected_element_ceiling: 1,
            expert_count: 3,
        }))
    };
    let outcome = eval(
        &graph,
        &valid_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut ceiling_role),
    );
    assert!(matches!(
        outcome,
        Err(EvalError::CastAuthority(ref boxed))
            if matches!(**boxed, CastAuthorityError::SelectedElementLimit { actual: 2, limit: 1, .. })
    ));

    let mut range_role = |value: poot_graph_ir::ValueId| {
        (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(ExactI32CastBounds {
            selected_element_ceiling: 2,
            expert_count: 3,
        }))
    };
    let error = eval(
        &graph,
        &HashMap::from([
            (table.id, exact_view(vec![3], [0, 16_777_217, 2])),
            (index.id, exact_view(vec![2], [0, 1])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut range_role),
    )
    .unwrap_err();
    // the exact-f32 integer range violation is now `EvalError::Cast(CastFault)`, not a
    // named `ExactI32Error` variant.
    assert!(matches!(
        &error,
        EvalError::Cast(fault) if fault.index == 1 && fault.value == crate::CastOperand::I32(16_777_217)
    ));

    let mut expert_role = |value: poot_graph_ir::ValueId| {
        (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(ExactI32CastBounds {
            selected_element_ceiling: 2,
            expert_count: 2,
        }))
    };
    let expert_error = eval(
        &graph,
        &valid_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut expert_role),
    )
    .unwrap_err();
    assert!(matches!(
        expert_error,
        EvalError::CastAuthority(ref boxed)
            if matches!(**boxed, CastAuthorityError::ExpertIdRange { index: 1, value: 2, experts: 2, .. })
    ));

    let mut accept_role = |value: poot_graph_ir::ValueId| {
        (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(ExactI32CastBounds {
            selected_element_ceiling: 2,
            expert_count: 3,
        }))
    };
    let result = eval(
        &graph,
        &valid_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut accept_role),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("expected selected f32")
    };
    assert_eq!(result.as_f32().unwrap(), &[0.0, 2.0]);
}

#[test]
fn exact_i32_errors_preserve_typed_sources() {
    use crate::EvalError;

    // Card 554d: `ExactI32Error::Graph` is gone with the whole exact-I32 lane; the unified `eval`
    // calls `g.validate()` itself before any binding or evaluation, so a malformed graph now surfaces
    // as the ordinary `EvalError::InvalidGraph` every other walk entry already used.
    let builder = Builder::new();
    let value = helpers::i32_constant(&builder, "value", vec![1]).unwrap();
    let mut graph = builder.finish(value);
    graph.output = usize::MAX;
    assert!(matches!(
        eval(
            &graph,
            &HashMap::from([(value.id, exact_view(vec![1], [1]))]),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        ),
        Err(EvalError::InvalidGraph(
            poot_graph_ir::GraphValidationError::OutputNotDefined { value: usize::MAX }
        ))
    ));
}

#[test]
fn poot_eval_does_not_depend_on_graph_plan() {
    // Cargo accepts this edge because `poot-graph-plan` only dev-depends on `poot-eval`, so the manifest is
    // checked directly.
    let manifest = include_str!("../../Cargo.toml");
    assert!(!manifest.contains("poot-graph-plan"));
}

#[test]
fn i32_widening_to_f32_is_exact_only_within_two_pow_24() {
    assert_eq!(
        &*HostTensor::i32(vec![2], vec![16_777_216, -16_777_216])
            .to_f32()
            .unwrap(),
        &[16_777_216.0, -16_777_216.0]
    );
    assert!(HostTensor::i32(vec![1], vec![16_777_217]).to_f32().is_err());
    assert!(
        HostTensor::i32(vec![1], vec![-16_777_217])
            .to_f32()
            .is_err()
    );
    assert!(HostTensor::i32(vec![1], vec![i32::MIN]).to_f32().is_err());
}

#[test]
fn exact_i32_cast_authority_set_but_silent_is_missing_authorization() {
    // Card 554d: this test used to be `exact_i32_rejects_generic_fallback`, proving the deleted
    // exact-I32 lane's "admitted op list" was a list, not a default (`Transpose` and `BinOp::Max` were
    // outside it) and exercising the deleted `eval_value` entry point's own narrower named-op-list
    // fast path. The unified walk has no such list: every op dispatches on its own operands, so
    // `ops::movement::transpose`/`ops::elementwise::binary_i32_word` (`BinOp::Max`)
    // evaluate I32 operands unconditionally now - there is no "generic fallback" left to reject. Both
    // rows are dropped (flagged in the migration report); `eval_value` is one of the eight deleted
    // entry points with no narrower-path successor to call instead.
    //
    // The one surviving check: a `Cast(I32 -> F32)` over a `Gather` is unauthorized when
    // `cast_authority` is configured but its closure returns `None` for this particular cast.
    use crate::EvalError;
    use crate::cast_authority::CastAuthorityError;

    let cast_builder = Builder::new();
    let table = helpers::i32_constant(&cast_builder, "table", vec![1]).unwrap();
    let index = helpers::i32_constant(&cast_builder, "index", vec![1]).unwrap();
    let selected = cast_builder.gather(table, 0, index);
    let cast = cast_builder.cast(selected, DType::F32);
    let cast_graph = cast_builder.finish(cast);
    let mut silent_authority = |_: poot_graph_ir::ValueId| None;
    let outcome = eval(
        &cast_graph,
        &HashMap::from([
            (table.id, exact_view(vec![1], [0])),
            (index.id, exact_view(vec![1], [0])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut silent_authority),
    );
    assert!(matches!(
        outcome,
        Err(EvalError::CastAuthority(ref boxed))
            if matches!(**boxed, CastAuthorityError::MissingCastAuthorization { cast: missing } if missing == cast.id)
    ));
}

#[test]
fn i32_boundary_test_is_scalar_bounded() {
    use crate::EvalError;
    use crate::cast_authority::CastAuthorityError;
    use crate::cast_authority::{ExactI32CastBounds, ExactI32CastRole};

    let builder = Builder::new();
    let table = helpers::i32_constant(&builder, "boundary.table", vec![2]).unwrap();
    let index = helpers::i32_constant(&builder, "boundary.index", vec![2]).unwrap();
    let selected = builder.gather(table, 0, index);
    let cast = builder.cast(selected, DType::F32);
    let graph = builder.finish(cast);
    let indices = exact_view(vec![2], [0, 1]);

    let run = |words: [i32; 2], experts: usize| {
        let mut role = |value: poot_graph_ir::ValueId| {
            (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(
                ExactI32CastBounds {
                    selected_element_ceiling: 2,
                    expert_count: experts,
                },
            ))
        };
        eval(
            &graph,
            &HashMap::from([
                (table.id, exact_view(vec![2], words)),
                (index.id, indices.clone()),
            ]),
            EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut role),
        )
    };
    assert!(run([16_777_216, 2], 16_777_217).is_ok());
    assert!(matches!(
        run([-16_777_216, 2], 3),
        Err(EvalError::CastAuthority(ref boxed))
            if matches!(**boxed, CastAuthorityError::ExpertIdRange { index: 0, value: -16_777_216, .. })
    ));
    // the exact-f32 integer range violation (checked before the expert-id range, so it
    // wins at word 0) is now `EvalError::Cast(CastFault)`, not a named `ExactI32Error` variant.
    assert!(matches!(
        &run([16_777_217, 2], 16_777_218),
        Err(EvalError::Cast(fault))
            if fault.index == 0 && fault.value == crate::CastOperand::I32(16_777_217)
    ));
    assert!(matches!(
        run([-1, 3], 3),
        Err(EvalError::CastAuthority(ref boxed))
            if matches!(**boxed, CastAuthorityError::ExpertIdRange { .. })
    ));
}

/// Coverage rows for the evaluator rejections this table must exercise: the kind of Cast source each
/// `CastSourceNot*` error says the input is not. Every other `CastAuthorityError` variant maps to
/// `None` explicitly, so a new variant fails to compile until it is classified here. (Card 554d: this
/// used to classify the deleted `ExactI32Error`'s variants at large; the three `CastSourceNot*` rows
/// now live on `cast_authority::CastAuthorityError` instead, which this now matches directly.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExactI32RejectionRow {
    Gather,
    GuardedGather,
    Observation,
}

fn exact_i32_rejection_row(
    error: &crate::cast_authority::CastAuthorityError,
) -> Option<ExactI32RejectionRow> {
    use crate::cast_authority::CastAuthorityError as E;
    match error {
        E::CastSourceNotGather { .. } => Some(ExactI32RejectionRow::Gather),
        E::CastSourceNotGuardedGather { .. } => Some(ExactI32RejectionRow::GuardedGather),
        E::CastSourceNotObservation { .. } => Some(ExactI32RejectionRow::Observation),
        E::MissingCastAuthorization { .. }
        | E::SelectedElementLimit { .. }
        | E::ExpertIdRange { .. } => None,
    }
}

#[test]
fn exact_i32_rejection_table_covers_cast_source() {
    use crate::EvalError;
    use crate::cast_authority::CastAuthorityError;
    use crate::cast_authority::{ExactI32CastBounds, ExactI32CastRole};

    let mut covered = std::collections::BTreeSet::new();

    // An authorized Cast whose input is not a Gather result rejects before any selected allocation.
    let builder = Builder::new();
    let left = helpers::i32_constant(&builder, "left", vec![2]).unwrap();
    let right = helpers::i32_constant(&builder, "right", vec![2]).unwrap();
    let difference = builder.binary(BinOp::Sub, left, right);
    let cast = builder.cast(difference, DType::F32);
    let graph = builder.finish(cast);
    let mut role = |value: poot_graph_ir::ValueId| {
        (value == cast.id).then_some(ExactI32CastRole::HostCheckedSelector(ExactI32CastBounds {
            selected_element_ceiling: 2,
            expert_count: 3,
        }))
    };
    let error = eval(
        &graph,
        &HashMap::from([
            (left.id, exact_view(vec![2], [2, 1])),
            (right.id, exact_view(vec![2], [1, 1])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut role),
    )
    .unwrap_err();
    let EvalError::CastAuthority(error) = error else {
        panic!("expected a cast-authority rejection, got {error:?}")
    };
    assert_eq!(
        exact_i32_rejection_row(&error),
        Some(ExactI32RejectionRow::Gather)
    );
    assert_eq!(
        *error,
        CastAuthorityError::CastSourceNotGather {
            cast: cast.id,
            input: difference.id,
        }
    );
    covered.insert(ExactI32RejectionRow::Gather);

    // Card 372c: the two new roles have their own tests below; record their rows so the coverage set
    // stays exhaustive over the cast-source rows of `ExactI32RejectionRow`.
    covered.insert(ExactI32RejectionRow::GuardedGather);
    covered.insert(ExactI32RejectionRow::Observation);

    assert_eq!(
        covered.into_iter().collect::<Vec<_>>(),
        [
            ExactI32RejectionRow::Gather,
            ExactI32RejectionRow::GuardedGather,
            ExactI32RejectionRow::Observation,
        ]
    );
}

// --- Card 372c: the in-graph bounds guard on the CPU oracle ---

/// The guarded selector chain on its own: a Gather over an exact I32 table, the canonical guard, the
/// selector Cast as the primary output, and the guard's flag as the only validation output.
fn guarded_selector_graph(
    upper_exclusive: usize,
) -> (
    poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
) {
    use poot_graph_ir::ValidationId;

    let builder = Builder::new();
    let table = helpers::i32_constant(&builder, "route.table", vec![4]).unwrap();
    let indices = helpers::i32_constant(&builder, "route.indices", vec![4]).unwrap();
    let raw = builder.gather(table, 0, indices);
    let guard = builder.guard_index_bounds(raw, upper_exclusive).unwrap();
    let selected = builder.cast(guard.guarded, DType::F32);
    let graph = helpers::finish_with_validations(
        builder,
        selected,
        &[(ValidationId(3), "route.bounds", guard.witness)],
    )
    .unwrap();
    (graph, table.id, indices.id, selected.id)
}

/// The authorization a guarded chain needs: the selector Cast is guarded, the flag Cast is a witness.
fn guarded_roles(
    selected: poot_graph_ir::ValueId,
) -> impl FnMut(poot_graph_ir::ValueId) -> Option<crate::cast_authority::ExactI32CastRole> {
    move |value| {
        Some(if value == selected {
            crate::cast_authority::ExactI32CastRole::GuardedSelector {
                selected_element_ceiling: 16_777_216,
            }
        } else {
            crate::cast_authority::ExactI32CastRole::Witness {
                selected_element_ceiling: 16_777_216,
            }
        })
    }
}

#[test]
fn guarded_selector_publishes_in_range_ids_unchanged_on_cpu() {
    use crate::Value;

    // Every id is in range, so the witness passes and the guarded selector publishes the ids unchanged
    // (the identity half of the guard; goes red if the `Select` returned the zero branch for a valid
    // lane). The clamp is only observable on device (`wgpu_372c_guard_clamps_out_of_range_lanes`),
    // because CPU publication is refused as soon as any lane is out of range.
    let (graph, table, indices, selected) = guarded_selector_graph(4);
    let value = eval(
        &graph,
        &HashMap::from([
            (table, exact_view(vec![4], [3, 0, 2, 1])),
            (indices, exact_view(vec![4], [0, 1, 2, 3])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut guarded_roles(selected)),
    )
    .unwrap()
    .output;
    let Value::Host(tensor) = value else {
        panic!("a selector Cast publishes a dense f32 tensor")
    };
    assert_eq!(tensor.as_f32().unwrap(), [3.0, 0.0, 2.0, 1.0]);
}

/// [`guarded_selector_graph`] with the Gather replaced by arithmetic, so the guard is canonical but what
/// it clamps is not a selector.
fn guarded_non_gather_graph() -> (
    poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
) {
    use poot_graph_ir::ValidationId;

    let builder = Builder::new();
    let left = helpers::i32_constant(&builder, "route.left", vec![4]).unwrap();
    let right = helpers::i32_constant(&builder, "route.right", vec![4]).unwrap();
    let raw = builder.binary(BinOp::Sub, left, right);
    let guard = builder.guard_index_bounds(raw, 4).unwrap();
    let selected = builder.cast(guard.guarded, DType::F32);
    let graph = helpers::finish_with_validations(
        builder,
        selected,
        &[(ValidationId(3), "route.bounds", guard.witness)],
    )
    .unwrap();
    (graph, left.id, right.id, selected.id, guard.guarded.id)
}

#[test]
fn guarded_selector_role_requires_the_guard_to_clamp_a_gather() {
    use crate::EvalError;
    use crate::cast_authority::CastAuthorityError;

    // Mirror of the planner's `authorization_rejects_a_guard_over_something_that_is_not_a_gather`: a
    // canonical guard clamping arithmetic is not a selector chain (card 371 authorizes a Cast over a
    // Gather result), and the oracle must be as strict as the planner rather than lean on the exact-f32
    // range.
    let (graph, left, right, selected, guarded) = guarded_non_gather_graph();
    let error = eval(
        &graph,
        &HashMap::from([
            (left, exact_view(vec![4], [3, 2, 1, 0])),
            (right, exact_view(vec![4], [0, 0, 0, 0])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut guarded_roles(selected)),
    )
    .unwrap_err();
    let EvalError::CastAuthority(error) = error else {
        panic!("expected a cast-authority rejection, got {error:?}")
    };
    assert_eq!(
        *error,
        CastAuthorityError::CastSourceNotGuardedGather {
            cast: selected,
            input: guarded,
        }
    );
}

#[test]
fn guarded_selector_with_a_forged_bound_cannot_publish_inexact_lanes() {
    use crate::EvalError;

    // A canonical guard over a Gather whose bound is 2^25, not a real expert count. Nothing checks the
    // bound (`validate_cast_source` reads the shape; the role carries no expert count), so the clamp
    // admits 2^24 + 1, which an f32 cannot hold. The witness passes (no lane reaches 2^25), so the
    // exact-f32 lane range is the only guard against a published "exact" Cast with inexact lanes.
    let (graph, table, indices, selected) = guarded_selector_graph(1 << 25);
    let result = eval(
        &graph,
        &HashMap::from([
            (table, exact_view(vec![4], [16_777_217, 0, 0, 0])),
            (indices, exact_view(vec![4], [0, 1, 2, 3])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut guarded_roles(selected)),
    );
    // the exact-f32 integer range violation is now `EvalError::Cast(CastFault)`, not a
    // named `ExactI32Error` variant.
    match result {
        Err(EvalError::Cast(fault)) => {
            assert_eq!(fault.value, crate::CastOperand::I32(16_777_217));
        }
        other => panic!("a forged bound must be caught, got {other:?}"),
    }
}

#[test]
fn witness_counts_exactly_the_out_of_range_lanes() {
    use crate::EvalError;

    // Table lanes -1 (negative), 4 (at the bound) and 9 (above it) are out of range; 1 is not. Expected
    // packet bits are the literal count, computed by the test, not the production reduce.
    let (graph, table, indices, selected) = guarded_selector_graph(4);
    let error = eval(
        &graph,
        &HashMap::from([
            (table, exact_view(vec![4], [-1, 4, 9, 1])),
            (indices, exact_view(vec![4], [0, 1, 2, 3])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut guarded_roles(selected)),
    )
    .unwrap_err();
    let EvalError::Validation(failure) = error else {
        panic!("an out-of-range id must fail as a neutral validation failure, got {error:?}")
    };
    assert_eq!(failure.id, poot_graph_ir::ValidationId(3));
    assert_eq!(failure.name, "route.bounds");
    assert_eq!(failure.lane, 0);
    assert_eq!(failure.observed_bits, 3.0f32.to_bits());
}

#[test]
fn cpu_exact_eval_checks_the_packet_before_publishing() {
    use crate::EvalError;

    // The guard clamps the bad lane so the primary value is computable, but publication must still be
    // refused: a published value would silently route every bad token to expert 0.
    let (graph, table, indices, selected) = guarded_selector_graph(4);
    let result = eval(
        &graph,
        &HashMap::from([
            (table, exact_view(vec![4], [0, 1, 2, 7])),
            (indices, exact_view(vec![4], [0, 1, 2, 3])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut guarded_roles(selected)),
    );
    assert!(
        matches!(result, Err(EvalError::Validation(_))),
        "a failing witness must publish nothing, got {result:?}"
    );

    // The same graph with every lane in range publishes, so the gate is not always closed.
    assert!(
        eval(
            &graph,
            &HashMap::from([
                (table, exact_view(vec![4], [0, 1, 2, 3])),
                (indices, exact_view(vec![4], [0, 1, 2, 3])),
            ]),
            EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut guarded_roles(selected)),
        )
        .is_ok()
    );
}

/// The `Sub`-fed Cast of [`exact_i32_rejection_table_covers_owner_facts_and_cast_source`], reused by the
/// two card 372c role tests: an authorized Cast over an unauthorized chain.
fn unauthorized_cast_graph() -> (
    poot_graph_ir::Graph,
    HashMap<poot_graph_ir::ValueId, crate::Value>,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
) {
    let builder = Builder::new();
    let left = helpers::i32_constant(&builder, "left", vec![2]).unwrap();
    let right = helpers::i32_constant(&builder, "right", vec![2]).unwrap();
    let difference = builder.binary(BinOp::Sub, left, right);
    let cast = builder.cast(difference, DType::F32);
    let graph = builder.finish(cast);
    let inputs = HashMap::from([
        (left.id, exact_view(vec![2], [2, 1])),
        (right.id, exact_view(vec![2], [1, 1])),
    ]);
    (graph, inputs, cast.id, difference.id)
}

#[test]
fn guarded_selector_role_requires_a_canonical_guard() {
    use crate::EvalError;
    use crate::cast_authority::CastAuthorityError;
    use crate::cast_authority::ExactI32CastRole;

    // Naming the guarded role does not grant its weaker lane checks: the evaluator looks for the guard
    // in the graph and finds a plain Sub.
    let (graph, inputs, cast, difference) = unauthorized_cast_graph();
    let mut role = |value: poot_graph_ir::ValueId| {
        (value == cast).then_some(ExactI32CastRole::GuardedSelector {
            selected_element_ceiling: 2,
        })
    };
    let error = eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut role),
    )
    .unwrap_err();
    let EvalError::CastAuthority(error) = error else {
        panic!("expected a cast-authority rejection, got {error:?}")
    };
    assert_eq!(
        *error,
        CastAuthorityError::CastSourceNotGuardedGather {
            cast,
            input: difference,
        }
    );
}

#[test]
fn witness_cast_role_requires_an_observation_source() {
    use crate::EvalError;
    use crate::cast_authority::CastAuthorityError;
    use crate::cast_authority::ExactI32CastRole;

    // The witness role is only for a flag an ordered comparison produced. A Sub is not one.
    let (graph, inputs, cast, difference) = unauthorized_cast_graph();
    let mut role = |value: poot_graph_ir::ValueId| {
        (value == cast).then_some(ExactI32CastRole::Witness {
            selected_element_ceiling: 2,
        })
    };
    let error = eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).cast_authority(&mut role),
    )
    .unwrap_err();
    let EvalError::CastAuthority(error) = error else {
        panic!("expected a cast-authority rejection, got {error:?}")
    };
    assert_eq!(
        *error,
        CastAuthorityError::CastSourceNotObservation {
            cast,
            input: difference,
        }
    );
}

// Card 383: the index arithmetic a flat-gather selector needs, and the rule keeping it index arithmetic.
// The shape under test is `Gather(table, 0, Reshape(Add(Mul(row, stride), slot)))`: ids stay I32, so
// the selected-only Cast of card 371 keeps a single consumer.

/// One `[T, k]` gather-address chain over a flat table, as the flat-gather selector builds it. Returns
/// the graph, the row and slot input ids, and the flat table id.
fn address_chain_graph(
    rows: usize,
    slots: usize,
    stride: i32,
    table_extent: usize,
) -> (
    poot_graph_ir::Graph,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
    poot_graph_ir::ValueId,
) {
    let builder = Builder::new();
    let table = helpers::i32_constant(&builder, "address.table", vec![table_extent]).unwrap();
    let row_ids = helpers::i32_constant(&builder, "address.rows", vec![rows]).unwrap();
    let slot_iota = helpers::i32_constant(&builder, "address.slots", vec![slots]).unwrap();
    let row_column = builder.reshape(row_ids, vec![rows, 1]);
    let row_base = builder.binary_scalar(BinOp::Mul, row_column, Scalar::I32(stride));
    let address = builder.binary(BinOp::Add, row_base, slot_iota);
    let flat = builder.reshape(address, vec![rows * slots]);
    let gathered = builder.gather(table, 0, flat);
    (builder.finish(gathered), row_ids.id, slot_iota.id, table.id)
}

#[test]
fn exact_i32_reshape_is_a_view_over_the_same_words() {
    use crate::{ExactI32TensorView, Value};

    // View half: the same word allocation under a new shape.
    let source =
        ExactI32TensorView::try_from_words(vec![6], Arc::from([7, 8, 9, 10, 11, 12])).unwrap();
    let reshaped = source.reshaped(vec![2, 3]).unwrap();
    assert_eq!(reshaped.shape(), [2, 3]);
    assert_eq!(reshaped.i32_words(), source.i32_words());
    assert!(Arc::ptr_eq(reshaped.word_owner(), source.word_owner()));
    // Card 554d: `ExactI32Error::ElementCount` is gone; `ExactI32TensorView::reshaped` now reports a
    // mismatched element count through the plain `EvalError::unsupported` helper every other
    // low-level view/word-count mismatch in this crate uses.
    let error = source.reshaped(vec![4]).unwrap_err();
    let crate::EvalError::Unsupported { op, detail, .. } = &error else {
        panic!("expected an Unsupported element-count mismatch, got {error:?}")
    };
    assert_eq!(*op, "exact_i32_view");
    assert!(
        detail.contains("I32 shape [4] expected 4 words, got 6"),
        "{detail}"
    );

    // Evaluator half: a reshaped address reads the rows its new shape names. Card 554d: Gather over an
    // exact-I32 table always publishes `Value::Host(HostTensor::i32(..))` now (never
    // `Value::Owner(ExactValue::I32(..))`); the deleted `ExactI32Observer` accounting
    // (`operation_derived_i32_allocations`/`ExactI32AllocationKind`) that used to prove the Reshape
    // itself derived no allocation has no replacement (flagged in the migration report).
    let (graph, rows, slots, table) = address_chain_graph(2, 3, 3, 6);
    let result = eval(
        &graph,
        &HashMap::from([
            (table, exact_view(vec![6], [10, 11, 12, 13, 14, 15])),
            (rows, exact_view(vec![2], [0, 1])),
            (slots, exact_view(vec![3], [0, 1, 2])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("expected a dense gather result, got {result:?}")
    };
    assert_eq!(result.shape(), vec![6]);
    assert_eq!(result.as_i32(), Some(&[10, 11, 12, 13, 14, 15][..]));
}

#[test]
fn exact_i32_reshape_accepts_a_dense_carrier_too() {
    use crate::Value;

    // Card 554d: this test used to be `exact_i32_reshape_refuses_a_non_authoritative_value`, proving
    // the deleted exact-I32 lane's own rule that `Reshape` required its input to be literally
    // `Value::Owner(ExactValue::I32(..))`, refusing a plain `Value::Host` I32 carrier (even with
    // authoritative `the I32 words`) as "non-authoritative". The unified walk has no such distinction:
    // `walk.rs`'s `i32_tensor` helper reads a dense-with-ints carrier and an exact-I32 view
    // identically (both are zero-copy `Arc` reads), so `Reshape` of either now evaluates to the same
    // result - there is nothing left to refuse. This row keeps the shape of the old one (same graph,
    // same inputs) but checks the surviving behavior: a dense carrier for `rows` computes the same
    // gather result as the all-exact-view binding above.
    let (graph, rows, slots, table) = address_chain_graph(2, 3, 3, 6);
    let result = eval(
        &graph,
        &HashMap::from([
            (table, exact_view(vec![6], [10, 11, 12, 13, 14, 15])),
            (rows, Value::from(i32_tensor(vec![2], &[0, 1]))),
            (slots, exact_view(vec![3], [0, 1, 2])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    assert_eq!(result.as_i32(), Some(&[10, 11, 12, 13, 14, 15][..]));
}

#[test]
fn exact_i32_index_arithmetic_is_wordwise_and_wrapping() {
    use crate::{EvalError, Value};

    // The table is twice as long as the addresses need, so a wrong stride shows up as the wrong row,
    // not a bounds rejection.
    //
    // Rows 1 and 2, not 0 and 1: a row base of zero makes `Add` interchangeable with `Sub` on the first
    // three lanes, and a negative address would die on the Gather bound rather than on the checked
    // value. With bases 3 and 6 every mutated address stays inside the table.
    let (graph, rows, slots, table) = address_chain_graph(2, 3, 3, 12);
    let table_words = exact_view(
        vec![12],
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    );
    let result = eval(
        &graph,
        &HashMap::from([
            (table, table_words.clone()),
            (rows, exact_view(vec![2], [1, 2])),
            (slots, exact_view(vec![3], [0, 1, 2])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("expected a dense gather result, got {result:?}")
    };
    // row * 3 + slot, for rows 1 and 2: an offset add of a stride multiply. Subtracting instead reads
    // [103, 102, 101, 106, 105, 104]; adding instead reads from rows 4 and 5.
    assert_eq!(result.as_i32(), Some(&[103, 104, 105, 106, 107, 108][..]));

    // The same chain with row 3 reads the second half of the table, which only the multiply reaches.
    let result = eval(
        &graph,
        &HashMap::from([
            (table, table_words.clone()),
            (rows, exact_view(vec![2], [3, 2])),
            (slots, exact_view(vec![3], [0, 1, 2])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("expected a dense gather result, got {result:?}")
    };
    assert_eq!(result.as_i32(), Some(&[109, 110, 111, 106, 107, 108][..]));

    // Wrapping, not saturating or checked: `i32::MAX * 2` is -2, which the Gather bound reports
    // verbatim; a saturating multiply would report `i32::MAX`.
    let (graph, rows, slots, table) = address_chain_graph(1, 1, 2, 12);
    let error = eval(
        &graph,
        &HashMap::from([
            (table, table_words),
            (rows, exact_view(vec![1], [i32::MAX])),
            (slots, exact_view(vec![1], [0])),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_err();
    // Card 555: `ExactI32Error::GatherIndex` is gone; the Gather bounds check now reports through the
    // one index rule's `EvalError::Index`.
    let EvalError::Index(fault) = &error else {
        panic!("expected a bounds rejection, got {error:?}")
    };
    assert_eq!(fault.position, 0);
    assert_eq!(fault.value, crate::IndexValue::I32(-2));
    assert_eq!(fault.len, 12);
    assert_eq!(fault.kind, crate::ops::index_rule::IndexFaultKind::Negative);
}

#[test]
fn computed_gather_addresses_are_bounds_checked() {
    use crate::{EvalError, Value};

    // The score arm of the flat-gather formulation: an authoritative I32 address into a dense f32 row.
    // The address is computed, so only this check stands between it and the table.
    let builder = Builder::new();
    let scores = builder.constant("scores", TensorType::f32(vec![6]));
    let row_ids = helpers::i32_constant(&builder, "score.rows", vec![2]).unwrap();
    let expert_ids = helpers::i32_constant(&builder, "score.ids", vec![2]).unwrap();
    let row_base = builder.binary_scalar(BinOp::Mul, row_ids, Scalar::I32(3));
    let address = builder.binary(BinOp::Add, row_base, expert_ids);
    let gathered = builder.gather(scores, 0, address);
    let graph = builder.finish(gathered);
    let bind = |ids: [i32; 2]| {
        HashMap::from([
            (
                scores.id,
                Value::Host(HostTensor::f32(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            ),
            (row_ids.id, exact_view(vec![2], [0, 1])),
            (expert_ids.id, exact_view(vec![2], ids)),
        ])
    };

    let result = eval(
        &graph,
        &bind([2, 0]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output;
    let Value::Host(result) = result else {
        panic!("a dense table gathers to a dense tensor")
    };
    assert_eq!(result.as_f32().unwrap(), [2.0, 3.0]);

    let error = eval(
        &graph,
        &bind([2, 3]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_err();
    let EvalError::Index(fault) = &error else {
        panic!("expected a bounds rejection, got {error:?}")
    };
    assert_eq!(fault.position, 1);
    assert_eq!(fault.value, crate::IndexValue::I32(6));
    assert_eq!(fault.len, 6);
    assert_eq!(
        fault.kind,
        crate::ops::index_rule::IndexFaultKind::OutOfRange
    );
}

/// A decode-shaped graph: one dense cache and one exact-I32 counter, both carried across the step.
fn state_carrying_graph() -> (
    poot_graph_ir::Graph,
    poot_graph_ir::Traced,
    poot_graph_ir::Traced,
) {
    let b = Builder::new();
    let cache = b.state_input("cache", TensorType::f32(vec![2]), StateRole::Recurrent);
    let counter = b.state_input(
        "counter",
        TensorType::new(vec![2], DType::I32),
        StateRole::Recurrent,
    );
    let cache_next = b.binary(BinOp::Add, cache, cache);
    let counter_next = b.binary_scalar(BinOp::Sub, counter, Scalar::I32(-1));
    let logits = b.binary(BinOp::Sub, cache_next, cache);
    let graph = b.finish_with_state(logits, &[(cache, cache_next), (counter, counter_next)]);
    (graph, cache, counter)
}

fn state_carrying_bindings(
    cache: poot_graph_ir::Traced,
    counter: poot_graph_ir::Traced,
) -> HashMap<usize, crate::Value> {
    HashMap::from([
        (
            cache.id,
            crate::Value::Host(HostTensor::f32(vec![2], vec![3.0, -1.0])),
        ),
        (counter.id, exact_view(vec![2], [5, 7])),
    ])
}

#[test]
fn exact_i32_with_state_publishes_every_pair_in_graph_order() {
    use crate::Value;

    let (graph, cache, counter) = state_carrying_graph();
    let inputs = state_carrying_bindings(cache, counter);
    let evaluation = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    let (output, state) = (evaluation.output, evaluation.state);

    let Value::Host(output) = output else {
        panic!("the primary output is dense")
    };
    assert_eq!(output.as_f32().unwrap(), &[3.0, -1.0]);
    assert_eq!(state.len(), 2, "one value per state pair");
    // Pair order is the graph's. Card 554d: an I32 equation (`counter_next` is a `Binary(Sub)`) now
    // always publishes `Value::Host(HostTensor::i32(..))`, never `Value::Owner(ExactValue::I32(..))` -
    // and `check_state_storage` would refuse the exact carrier for I32 state even if it appeared (see
    // `exact_i32_with_state_requires_the_dense_carrier`), so both pairs publish dense now.
    let Value::Host(cache_out) = &state[0] else {
        panic!(
            "the f32 cache publishes on the dense lane, got {:?}",
            state[0]
        )
    };
    assert_eq!(cache_out.as_f32().unwrap(), &[6.0, -2.0]);
    let Value::Host(counter_out) = &state[1] else {
        panic!(
            "the I32 counter publishes on the dense lane, got {:?}",
            state[1]
        )
    };
    assert_eq!(counter_out.as_i32(), Some(&[6, 8][..]));
}

// Card 554d: `exact_i32_rejects_a_stateful_graph` is deleted, not migrated. It proved the deleted
// exact-I32 lane's own restriction that it could not publish state at all
// (`ExactI32Error::DiscardedState`); the unified `eval` has no such restriction - `Evaluation::state`
// is populated unconditionally for every graph with state pairs, stateful or not, as the test above
// and the rest of this file's state-carrying tests already demonstrate.

#[test]
fn exact_i32_with_state_requires_the_dense_carrier() {
    use crate::{EvalError, Value};

    // Card 554d: this test used to be named `..._rejects_a_state_value_that_left_the_exact_lane` and
    // proved the OPPOSITE of what the unified walk does now: the deleted exact-I32 lane required an
    // I32 state value to stay on the exact carrier, rejecting a plain `Value::Host` (even with
    // authoritative the I32 words) as having "left the exact lane". The unified walk's own
    // `check_state_storage` (`walk.rs`) does the reverse: its `carrier_matches` for `DType::I32` admits
    // only `Value::Host(_)`, with no arm for `Value::Owner(ExactValue::I32(_))` at all - so an I32
    // state value now round-trips only on the dense carrier, and a bound `Value::Owner` state value is
    // the one that is refused.
    let b = Builder::new();
    let counter = b.state_input(
        "counter",
        TensorType::new(vec![2], DType::I32),
        StateRole::Recurrent,
    );
    let cache = b.constant("cache", TensorType::f32(vec![2]));
    let logits = b.binary(BinOp::Add, cache, cache);
    let graph = b.finish_with_state(logits, &[(counter, counter)]);

    // The dense carrier now round-trips: an identity I32 state pair published on `Value::Host`.
    let dense_inputs = HashMap::from([
        (
            counter.id,
            Value::Host(HostTensor::i32(vec![2], vec![5, 7])),
        ),
        (
            cache.id,
            Value::Host(HostTensor::f32(vec![2], vec![1.0, 2.0])),
        ),
    ]);
    let state = eval(
        &graph,
        &dense_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .state;
    let Value::Host(counter_out) = &state[0] else {
        panic!("expected a dense I32 state value, got {:?}", state[0])
    };
    assert_eq!(counter_out.as_i32(), Some(&[5, 7][..]));

    // The exact carrier is now the one `check_state_storage` refuses for I32 state.
    let exact_inputs = HashMap::from([
        (counter.id, exact_view(vec![2], [5, 7])),
        (
            cache.id,
            Value::Host(HostTensor::f32(vec![2], vec![1.0, 2.0])),
        ),
    ]);
    let error = eval(
        &graph,
        &exact_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap_err();
    assert!(
        matches!(
            &error,
            EvalError::Input {
                value,
                got: "Value::Owner(ExactValue::I32)",
                ..
            } if *value == counter.id
        ),
        "{error:?}"
    );
}

#[test]
fn exact_i32_with_state_publishes_shared_allocations() {
    use crate::Value;

    // Identity state pairs, so the published value must be the caller's own allocation; a copy would
    // duplicate every cache/counter once per token.
    //
    // Card 554d: the exact-I32 half of this test used to bind `counter` on the exact carrier and prove
    // its `Arc<[i32]>` was shared by `Arc::ptr_eq`. The unified walk's `check_state_storage` now
    // refuses `Value::Owner(ExactValue::I32(_))` as an I32 state value altogether (see
    // `exact_i32_with_state_requires_the_dense_carrier`), so that half is migrated onto the dense
    // carrier instead - still zero-copy, just through `the I32 words`'s own `Arc` rather than
    // `ExactI32TensorView::word_owner`. The deleted `ExactI32Observer`'s "a published owner the
    // observer already accounted costs nothing to feed back" accounting assertion has no replacement
    // (flagged in the migration report).
    let b = Builder::new();
    let cache = b.state_input("cache", TensorType::f32(vec![2]), StateRole::Recurrent);
    let counter = b.state_input(
        "counter",
        TensorType::new(vec![2], DType::I32),
        StateRole::Recurrent,
    );
    let logits = b.binary(BinOp::Add, cache, cache);
    let graph = b.finish_with_state(logits, &[(cache, cache), (counter, counter)]);

    let cache_in = HostTensor::f32(vec![2], vec![3.0, -1.0]);
    let counter_in = HostTensor::i32(vec![2], vec![5, 7]);
    let inputs = HashMap::from([
        (cache.id, Value::Host(cache_in.clone())),
        (counter.id, Value::Host(counter_in.clone())),
    ]);
    let state = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .state;

    let Value::Host(cache_out) = &state[0] else {
        panic!("expected a dense cache, got {:?}", state[0])
    };
    assert_eq!(
        cache_out.as_f32().unwrap().as_ptr(),
        cache_in.as_f32().unwrap().as_ptr(),
        "the dense cache must be published, not copied"
    );
    let Value::Host(counter_out) = &state[1] else {
        panic!("expected a dense counter, got {:?}", state[1])
    };
    assert_eq!(
        counter_out.as_i32().unwrap().as_ptr(),
        counter_in.as_i32().unwrap().as_ptr(),
        "the I32 counter must be published, not copied"
    );
}

/// A one-cache decode step: write the update at the exact runtime position, carry the cache.
fn cache_write_graph() -> (
    poot_graph_ir::Graph,
    poot_graph_ir::Traced,
    poot_graph_ir::Traced,
    poot_graph_ir::Traced,
) {
    let b = Builder::new();
    let cache = b.state_input("cache", TensorType::f32(vec![4, 2]), StateRole::Recurrent);
    let update = b.constant("update", TensorType::f32(vec![1, 2]));
    let position = b.constant("position", TensorType::scalar(DType::I32));
    let next = b.dynamic_update_slice_dyn(cache, update, position, 0);
    let graph = b.finish_with_state(next, &[(cache, next)]);
    (graph, cache, update, position)
}

#[test]
fn exact_i32_dynamic_update_slice_writes_at_the_exact_position() {
    use crate::Value;

    let (graph, cache, update, position) = cache_write_graph();
    let mut carried = Value::Host(HostTensor::f32(vec![4, 2], vec![0.0; 8]));

    // Two steps, the second fed the first's published state: the write lands at its own row without
    // disturbing the previous row.
    for (step, row) in [(0usize, [9.0, -9.0]), (1, [4.0, -4.0])] {
        let inputs = HashMap::from([
            (cache.id, carried.clone()),
            (
                update.id,
                Value::Host(HostTensor::f32(vec![1, 2], row.to_vec())),
            ),
            (position.id, exact_view(vec![], [step as i32 + 1])),
        ]);
        let state = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .state;
        carried = state.into_iter().next().expect("one carried cache");
    }

    let Value::Host(carried) = carried else {
        panic!("the cache stays dense")
    };
    assert_eq!(
        carried.as_f32().unwrap(),
        &[0.0, 0.0, 9.0, -9.0, 4.0, -4.0, 0.0, 0.0],
        "each step must write its own row and leave the others alone"
    );
}

/// Card 554d: the deleted exact-I32 lane validated a runtime `DynamicUpdateSlice` index before writing
/// (`ExactI32Error::DynamicUpdateIndex`); the unified walk's dense `OpKind::DynamicUpdateSlice` arm
/// (`walk.rs`) resolves the runtime start through the one index rule
/// ([`crate::ops::index_rule::index_at`], card 555) before `ops::movement::dynamic_update_slice` runs,
/// for every dtype, not just the E4M3FN value path - so this fixture (a plain F32 cache/update) fails
/// closed through the same check rather than reaching an out-of-bounds write. A migration batch once
/// dropped this row with a comment claiming the dense arm had "no bounds check at all" and "both panic
/// rather than returning a typed `EvalError`"; that was true of an earlier draft, not of this commit's
/// `walk.rs` - restored.
///
/// Red mutation: delete the `index_at` resolution in `walk.rs`'s `DynamicUpdateSlice` arm (reinstate
/// the raw rounded index); row 4 of this 4-row cache panics ("index out of bounds: the len is 8 but
/// the index is 8" in `ops::movement::dynamic_update_slice`) - confirmed red, restored.
#[test]
fn exact_i32_dynamic_update_slice_rejects_an_out_of_range_position() {
    use crate::ops::index_rule::IndexFaultKind;
    use crate::{EvalError, Value};

    let (graph, cache, update, position) = cache_write_graph();
    let carried = Value::Host(HostTensor::f32(vec![4, 2], vec![0.0; 8]));

    for (bad_position, expected_kind) in [
        (4i32, IndexFaultKind::OutOfRange),
        (-1, IndexFaultKind::Negative),
    ] {
        let inputs = HashMap::from([
            (cache.id, carried.clone()),
            (
                update.id,
                Value::Host(HostTensor::f32(vec![1, 2], vec![1.0, -1.0])),
            ),
            (position.id, exact_view(vec![], [bad_position])),
        ]);
        let err = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect_err("an out-of-range position must fail closed");
        assert!(
            matches!(&err, EvalError::Index(fault) if fault.kind == expected_kind),
            "position {bad_position}: unexpected error {err:?}"
        );
    }
}

// Card 554d: `eval_value_i32_fast_path_is_a_named_op_list` is deleted, not migrated.
// `eval_value`/`eval_value_with_state` are two of the eight deleted entry points, and with them the
// separate narrower "storage-aware value path" this test proved had its own named-op-list admission
// (Binary/Select evaluate I32, Unary Not does not). The one unified walk has no such list: every I32
// op dispatches on its own operands uniformly, and `ops::elementwise::unary_i32_word` does define
// `UnOp::Not` (`!value`) - so under `eval`, an I32 Unary Not now evaluates successfully instead of
// failing closed. There is no narrower path left to prove has a shorter admitted-op list than the
// walk at large.

/// Card 449 (surviving half): an I32 input bound as a `Value::Host` that is not an I32
/// tensor fails closed. Card 554d: migrated off the deleted `eval_value_with_state` (and its E4M3FN
/// trick, which existed only to force that function's separate storage-aware path) onto the one
/// unified `eval`, which enforces the same rule at ordinary bind-time preflight
/// (`walk::preflight_bindings`) - no trick needed. Card 721: the rule is the strict bind contract
/// (a host input is exactly the declared dtype), so an F32 tensor is refused as a different dtype.
#[test]
fn eval_i32_requires_an_i32_tensor() {
    use crate::{EvalError, Value};

    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![2], DType::I32));
    let y = b.constant("y", TensorType::new(vec![2], DType::I32));
    let out = b.binary(BinOp::Add, x, y);
    let g = b.finish(out);
    let f32_only = HostTensor::f32(vec![2], vec![1.0, 2.0]);
    let inputs = HashMap::from([
        (x.id, Value::Host(f32_only)),
        (y.id, Value::Host(HostTensor::i32(vec![2], vec![1, 2]))),
    ]);
    let err = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect_err("an I32 input that is not an I32 tensor must fail closed");
    assert!(
        matches!(
            &err,
            EvalError::Input {
                got: "Value::Host of a different dtype",
                ..
            }
        ),
        "unexpected input error: {err:?}"
    );
}
