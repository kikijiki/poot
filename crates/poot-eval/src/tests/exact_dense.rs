use poot_tensor::DType;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use poot_graph_ir::{BinOp, Builder, RedOp, Storage, TensorType, ValueId};
use poot_load::packed_safetensors::{
    AuthenticatedInventory, AuthenticatedSafetensorsHandleSet, ExactSourceOwner,
    ExactSourceOwnerCache, InventoryDecision, PackedArtifactManifest, PackedOwnerCache,
    PackedSafetensorsLimits, PackedShardManifest, TensorDisposition, sha256_digest,
};

use super::allocation::allocated_bytes;
use super::exact_i32::authenticated_source_owner;
use crate::exact_dense::{DenseOwnerTensorView, ExactDenseError, bf16_word, decode_bf16_word};
use crate::{EvalBudget, EvalError, EvalOptions, ExactValue, Value, eval};
use poot_tensor::HostTensor;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

type UnaryGraphOperation = fn(&Builder, poot_graph_ir::Traced) -> poot_graph_ir::Traced;

const F32_BYTES: &[u8] = &[
    0x00, 0x00, 0x80, 0x3f, // 1.0
    0x00, 0x00, 0x00, 0x00, // +0
    0x00, 0x00, 0x00, 0x80, // -0
    0x45, 0x23, 0xc1, 0x7f, // positive NaN with payload
    0x45, 0x23, 0xc1, 0xff, // negative NaN with payload
    0x00, 0x00, 0x80, 0x7f, // +infinity
];
const BF16_BYTES: &[u8] = &[
    0x80, 0x3f, // 1.0
    0x00, 0x00, // +0
    0x00, 0x80, // -0
    0xc1, 0x7f, // positive NaN with payload
    0xc1, 0xff, // negative NaN with payload
    0x80, 0x7f, // +infinity
];

struct TempFixture {
    path: PathBuf,
}

impl TempFixture {
    fn new() -> Self {
        let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "poot-exact-dense-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write_fixture(path: &Path) -> (PackedArtifactManifest, PackedSafetensorsLimits) {
    let config = b"{}";
    fs::write(path.join("config.json"), config).unwrap();

    let header = br#"{"bf16":{"dtype":"BF16","shape":[6],"data_offsets":[0,12]},"f32":{"dtype":"F32","shape":[6],"data_offsets":[12,36]}}"#;
    let mut shard = Vec::new();
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(header);
    shard.extend_from_slice(BF16_BYTES);
    shard.extend_from_slice(F32_BYTES);
    let shard_name = "model.safetensors";
    fs::write(path.join(shard_name), &shard).unwrap();

    let index = br#"{"weight_map":{"bf16":"model.safetensors","f32":"model.safetensors"}}"#;
    fs::write(path.join("model.safetensors.index.json"), index).unwrap();
    (
        PackedArtifactManifest {
            repository: "local/exact-dense-fixture".to_string(),
            revision: "card370".to_string(),
            config_length: config.len(),
            config_sha256: sha256_digest(config),
            index_sha256: sha256_digest(index),
            shards: vec![PackedShardManifest {
                filename: shard_name.to_string(),
                file_length: shard.len(),
                file_sha256: sha256_digest(&shard),
                header_length: header.len(),
                header_sha256: sha256_digest(header),
            }],
        },
        PackedSafetensorsLimits {
            config_bytes: 1024,
            index_bytes: 1024,
            header_bytes_per_shard: 1024,
            shard_count: 1,
            tensor_entries: 2,
            selected_source_bytes: 36,
            packed_source_bytes: 1,
        },
    )
}

fn dense_decisions(
    inventory: AuthenticatedInventory<'_>,
) -> Result<Vec<InventoryDecision>, &'static str> {
    inventory
        .rows()
        .map(|row| {
            let disposition = match row.name() {
                "bf16" => TensorDisposition::DenseBf16,
                "f32" => TensorDisposition::DenseF32,
                _ => return Err("unexpected fixture row"),
            };
            Ok(InventoryDecision::new(row.key(), disposition))
        })
        .collect()
}

fn exact_owners() -> (Arc<ExactSourceOwner>, Arc<ExactSourceOwner>) {
    let fixture = TempFixture::new();
    let (manifest, limits) = write_fixture(&fixture.path);
    let mut authenticated =
        AuthenticatedSafetensorsHandleSet::authenticate(&fixture.path, manifest, limits).unwrap();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let result = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, dense_decisions)
        .unwrap();
    let find = |name: &str| Arc::clone(&result.exact_metadata.get(name).unwrap().owner);
    (find("bf16"), find("f32"))
}

fn views() -> (DenseOwnerTensorView, DenseOwnerTensorView) {
    let (bf16, f32) = exact_owners();
    (
        DenseOwnerTensorView::new(bf16).unwrap(),
        DenseOwnerTensorView::new(f32).unwrap(),
    )
}

#[test]
fn bf16_decode_uses_little_endian_bits() {
    let cases = [
        ([0x80, 0x3f], 0x3f80_0000),
        ([0x00, 0x80], 0x8000_0000),
        ([0xc1, 0x7f], 0x7fc1_0000),
        ([0x80, 0x7f], 0x7f80_0000),
    ];
    for (bytes, expected) in cases {
        assert_eq!(decode_bf16_word(bf16_word(bytes)).to_bits(), expected);
    }
}

#[test]
fn dense_view_keeps_exact_owner() {
    let (bf16_owner, f32_owner) = exact_owners();
    let bf16 = DenseOwnerTensorView::new(Arc::clone(&bf16_owner)).unwrap();
    let f32 = DenseOwnerTensorView::new(Arc::clone(&f32_owner)).unwrap();
    assert!(Arc::ptr_eq(bf16.owner(), &bf16_owner));
    assert!(Arc::ptr_eq(f32.owner(), &f32_owner));
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(f32.dtype(), DType::F32);
    assert_eq!(bf16.source_byte_len(), 12);
    assert_eq!(f32.source_byte_len(), 24);
    assert_eq!(bf16.descriptor().name(), "bf16");
    assert_eq!(f32.artifact().revision(), "card370");

    let reshaped = bf16.reshape(vec![2, 3]).unwrap();
    assert!(Arc::ptr_eq(reshaped.owner(), &bf16_owner));
    assert_eq!(reshaped.shape(), [2, 3]);
    assert!(matches!(
        bf16.reshape(vec![5]),
        Err(ExactDenseError::ReshapeElementCount { .. })
    ));

    let (distinct_bf16_owner, _) = exact_owners();
    let distinct = DenseOwnerTensorView::new(distinct_bf16_owner).unwrap();
    assert_eq!(bf16.owner().bytes(), distinct.owner().bytes());
    assert_ne!(bf16, distinct, "equal bytes are not owner identity");
}

#[test]
fn exact_value_dense_surface_preserves_card370_behavior() {
    let (bf16, f32) = views();
    let bf16_value = ExactValue::from(bf16.clone());
    let f32_value = ExactValue::from(f32.clone());

    assert_eq!(bf16_value.dtype(), DType::BF16);
    assert_eq!(bf16_value.shape(), [6]);
    assert_eq!(bf16_value.numel(), 6);
    assert_eq!(bf16_value.physical_bytes(), 12);
    assert_eq!(f32_value.dtype(), DType::F32);
    assert_eq!(f32_value.physical_bytes(), 24);
    assert_eq!(bf16_value, ExactValue::from(bf16.clone()));

    let reshaped = Value::Owner(bf16_value).reshape(vec![2, 3]).unwrap();
    assert_eq!(reshaped.shape().as_ref(), [2, 3]);
    assert_eq!(reshaped, Value::from(bf16.reshape(vec![2, 3]).unwrap()));
}

/// Allocator bytes to view, reshape, bind, evaluate as an identity, and account one BF16 owner.
fn dense_view_lifecycle_bytes(elements: usize) -> usize {
    // The fixture helper names its single tensor `i64` whatever its dtype.
    let owner = authenticated_source_owner(
        "i64",
        "BF16",
        elements,
        &vec![0; elements * 2],
        TensorDisposition::DenseBf16,
    );
    let builder = Builder::new();
    let constant = builder.constant("i64", TensorType::new([elements], DType::BF16));
    let graph = builder.finish(constant);
    let (output, bytes) = allocated_bytes(|| {
        let view = DenseOwnerTensorView::new(Arc::clone(&owner)).unwrap();
        let _reshaped = view.reshape(vec![2, elements / 2]).unwrap();
        let bound = HashMap::from([(constant.id, Value::from(view.clone()))]);
        eval(&graph, &bound, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
    });
    let Value::Owner(ExactValue::Dense(output)) = output else {
        panic!("identity output changed exact-dense storage");
    };
    assert!(Arc::ptr_eq(output.owner(), &owner));
    bytes
}

#[test]
fn dense_view_lifecycle_allocates_independent_of_source_size() {
    // A view is the owner plus metadata. No step may copy or decode the source, so a 64 KiB owner costs
    // exactly the bookkeeping bytes of an 8-element owner.
    assert_eq!(
        dense_view_lifecycle_bytes(1 << 15),
        dense_view_lifecycle_bytes(8)
    );
}

fn assert_graph_consumer(error: EvalError, expected: &str, value_id: ValueId) {
    let EvalError::ExactDense(ExactDenseError::GraphConsumer {
        value_id: actual_id,
        dtype,
        shape,
        storage,
        consumer,
    }) = error
    else {
        panic!("expected typed exact-dense graph-consumer error");
    };
    assert_eq!(actual_id, value_id);
    assert_eq!(dtype, DType::BF16);
    assert_eq!(shape, [2, 3]);
    assert_eq!(storage, Storage::Const);
    assert!(
        consumer.contains(expected),
        "{consumer:?} did not name {expected}"
    );
}

#[test]
fn dense_exact_value_rejects_nonview_ops() {
    let (bf16, _) = views();
    let view = bf16.reshape(vec![2, 3]).unwrap();
    let value = Value::from(view.clone());
    let index = HostTensor::f32(Vec::new(), vec![0.0]);
    assert!(matches!(
        value.transpose(&[1, 0]),
        Err(EvalError::ExactDense(_))
    ));
    assert!(matches!(
        value.slice(0, 0, 1),
        Err(EvalError::ExactDense(_))
    ));
    assert!(matches!(
        value.broadcast(vec![2, 2, 3]),
        Err(EvalError::ExactDense(_))
    ));
    assert!(matches!(
        value.gather(0, &index),
        Err(EvalError::ExactDense(_))
    ));
    assert!(matches!(
        Value::concat(&[&value], 0),
        Err(EvalError::ExactDense(_))
    ));
    assert!(matches!(
        value.dynamic_update_slice(&value, 0, 0),
        Err(EvalError::ExactDense(_))
    ));
    assert!(matches!(
        value.scatter_update(&value, &index, 0),
        Err(EvalError::ExactDense(_))
    ));

    // Spec 376 admits reshape, transpose, the widening cast, gather, and the BF16 weight of an F32 matmul on BF16
    // owners. Everything else still rejects, including a matmul of two BF16 operands.
    let graph_cases: [(&str, UnaryGraphOperation); 5] = [
        ("slice", |b, x| b.slice(x, 0, 0, 1)),
        ("matmul", |b, x| b.matmul(x, b.transpose(x, vec![1, 0]))),
        ("broadcast", |b, x| b.broadcast(x, vec![1, 2, 3])),
        ("add", |b, x| b.binary(BinOp::Add, x, x)),
        ("reduce", |b, x| b.reduce(RedOp::Sum, x, 1, false)),
    ];
    for (consumer, operation) in graph_cases {
        let builder = Builder::new();
        let source = builder.constant("bf16", TensorType::new([2, 3], DType::BF16));
        let output = operation(&builder, source);
        let graph = builder.finish(output);
        let error = eval(
            &graph,
            &HashMap::from([(source.id, value.clone())]),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap_err();
        assert_graph_consumer(error, consumer, source.id);
    }
}

#[test]
fn dense_eval_preflight_precedes_execution() {
    const PADDING_VALUES: usize = 2048;
    const UNRELATED_ELEMENTS: usize = 1 << 16;

    let (bf16, _) = views();
    let builder = Builder::new();
    let padding: Vec<ValueId> = (0..PADDING_VALUES)
        .map(|index| {
            builder
                .constant(&format!("padding.{index}"), TensorType::f32([1]))
                .id
        })
        .collect();
    let unrelated = builder.constant("unrelated", TensorType::f32([UNRELATED_ELEMENTS]));
    let exact = builder.constant("bf16", TensorType::new([2, 3], DType::BF16));
    let _allocating_prefix = builder.binary(BinOp::Add, unrelated, unrelated);
    let output = builder.slice(exact, 0, 0, 1);
    let graph = builder.finish(output);
    let mut inputs = HashMap::from([
        (
            unrelated.id,
            Value::Host(HostTensor::f32(
                vec![UNRELATED_ELEMENTS],
                vec![1.0; UNRELATED_ELEMENTS],
            )),
        ),
        (exact.id, Value::from(bf16.reshape(vec![2, 3]).unwrap())),
    ]);
    // Every graph input must be bound now (`walk::preflight_bindings`'s own MissingInput check, SC-010),
    // so the padding consts (here purely to inflate `g.values.len()`, never read) need a binding too -
    // otherwise MissingInput fires before the exact-dense GraphConsumer check this test targets.
    for id in padding {
        inputs.insert(id, Value::Host(HostTensor::f32(vec![1], vec![0.0])));
    }

    let (error, bytes) = allocated_bytes(|| {
        eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap_err()
    });
    assert_graph_consumer(error, "slice", exact.id);
    // The rejection must come before the per-value environment, any input copy, and the prefix equation's
    // output. Each of those alone is larger than this bound.
    let environment = PADDING_VALUES * std::mem::size_of::<Option<Value>>();
    assert!(
        bytes < environment,
        "exact-dense preflight allocated {bytes} bytes; the evaluator environment is {environment}"
    );
}

#[test]
fn dense_eval_identity_reuses_typed_metadata_validation() {
    let (bf16, _) = views();
    let builder = Builder::new();
    let exact = builder.constant("wrong", TensorType::new([6], DType::BF16));
    let graph = builder.finish(exact);
    let error = eval(
        &graph,
        &HashMap::from([(exact.id, Value::from(bf16.clone()))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output)
    .unwrap_err();
    assert!(matches!(
        error,
        EvalError::ExactDense(ExactDenseError::NameMismatch { value_id, .. })
            if value_id == exact.id
    ));

    let builder = Builder::new();
    let exact = builder.constant("bf16", TensorType::new([6], DType::BF16));
    let graph = builder.finish(exact);
    let output = eval(
        &graph,
        &HashMap::from([(exact.id, Value::from(bf16.clone()))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output)
    .unwrap();
    let Value::Owner(ExactValue::Dense(output)) = output else {
        panic!("identity output changed exact-dense storage");
    };
    assert!(Arc::ptr_eq(output.owner(), bf16.owner()));
}
