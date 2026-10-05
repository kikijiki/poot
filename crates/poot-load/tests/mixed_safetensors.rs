use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use poot_load::packed_safetensors::{
    AuthenticatedInventory, AuthenticatedInventoryRowKey, AuthenticatedSafetensorsHandleSet,
    ExactSourceKind, ExactSourceOwnerCache, InventoryDecision, MixedLoadError, MixedSourceCategory,
    PackedArtifactManifest, PackedOwnerCache, PackedSafetensorsError, PackedSafetensorsLimits,
    PackedShardManifest, SourceSpan, TensorDisposition, sha256_digest,
};
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::weights::WeightEntry;
use poot_quant::{OperandRole, PackedPayload, SourceRole};
use serde_json::{Value, json};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

const INVENTORY_DECISION_NEW: fn(
    AuthenticatedInventoryRowKey,
    TensorDisposition,
) -> InventoryDecision = InventoryDecision::new;

fn inventory_decision_parts(
    decision: InventoryDecision,
) -> (AuthenticatedInventoryRowKey, TensorDisposition) {
    let InventoryDecision { key, disposition } = decision;
    (key, disposition)
}

const SHARD_A: &str = "model-00001-of-00003.safetensors";
const SHARD_B: &str = "model-00002-of-00003.safetensors";
const SHARD_C: &str = "model-00003-of-00003.safetensors";
const CONFIG_BYTES: usize = 26;
const INDEX_BYTES: usize = 473;
const HEADER_A_BYTES: usize = 220;
const HEADER_B_BYTES: usize = 212;
const HEADER_C_BYTES: usize = 188;
const HEADER_TOTAL_BYTES: usize = 620;
// Card 544: config and index are admitted once, at authentication; a shard's full content is
// admitted once, at publication - two disjoint passes, not the same total read twice.
const CONFIG_INDEX_HASH_IO_BYTES: usize = CONFIG_BYTES + INDEX_BYTES;
const SHARD_HASH_IO_BYTES: usize = 733;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "poot-mixed-safetensors-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct TensorSource {
    name: &'static str,
    dtype: &'static str,
    shape: &'static [usize],
    bytes: Vec<u8>,
}

struct WrittenShard {
    manifest: PackedShardManifest,
    spans: BTreeMap<String, SourceSpan>,
    data_start: usize,
}

fn write_shard(root: &Path, filename: &str, tensors: Vec<TensorSource>) -> WrittenShard {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut spans = BTreeMap::new();
    for tensor in tensors {
        let start = data.len();
        data.extend_from_slice(&tensor.bytes);
        let end = data.len();
        spans.insert(tensor.name.to_string(), SourceSpan::new(start, end));
        header.insert(
            tensor.name.to_string(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(root.join(filename), &bytes).unwrap();
    WrittenShard {
        manifest: PackedShardManifest {
            filename: filename.to_string(),
            file_length: bytes.len(),
            file_sha256: sha256_digest(&bytes),
            header_length: header.len(),
            header_sha256: sha256_digest(&header),
        },
        spans,
        data_start: 8 + header.len(),
    }
}

fn write_at(path: &Path, offset: usize, byte: u8) {
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(offset as u64)).unwrap();
    file.write_all(&[byte]).unwrap();
    file.flush().unwrap();
}

fn refresh_shard_manifest(root: &Path, filename: &str) -> PackedShardManifest {
    let bytes = fs::read(root.join(filename)).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    PackedShardManifest {
        filename: filename.to_string(),
        file_length: bytes.len(),
        file_sha256: sha256_digest(&bytes),
        header_length,
        header_sha256: sha256_digest(&bytes[8..8 + header_length]),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ClassifierError {
    Sentinel { code: u32 },
    UnexpectedInventory,
    UnexpectedMetadata { tensor: String },
}

impl fmt::Display for ClassifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ClassifierError {}

struct Fixture {
    root: TempDir,
    manifest: PackedArtifactManifest,
    limits: PackedSafetensorsLimits,
    data_starts: BTreeMap<String, usize>,
    spans: BTreeMap<String, SourceSpan>,
}

struct ExpectedTensor {
    name: &'static str,
    shard: &'static str,
    span: SourceSpan,
    dtype: &'static str,
    shape: &'static [usize],
}

const EXPECTED_TENSORS: [ExpectedTensor; 9] = [
    ExpectedTensor {
        name: "dense.bf16",
        shard: SHARD_B,
        span: SourceSpan::new(40, 44),
        dtype: "BF16",
        shape: &[2],
    },
    ExpectedTensor {
        name: "dense.f32",
        shard: SHARD_C,
        span: SourceSpan::new(0, 8),
        dtype: "F32",
        shape: &[2],
    },
    ExpectedTensor {
        name: "dense.i64",
        shard: SHARD_C,
        span: SourceSpan::new(8, 24),
        dtype: "I64",
        shape: &[2],
    },
    ExpectedTensor {
        name: "layer.fp4.scale",
        shard: SHARD_B,
        span: SourceSpan::new(36, 40),
        dtype: "F8_E8M0",
        shape: &[2, 2],
    },
    ExpectedTensor {
        name: "layer.fp4.weight",
        shard: SHARD_B,
        span: SourceSpan::new(0, 36),
        dtype: "I8",
        shape: &[2, 18],
    },
    ExpectedTensor {
        name: "layer.fp8.scale",
        shard: SHARD_A,
        span: SourceSpan::new(10, 14),
        dtype: "F32",
        shape: &[1, 1],
    },
    ExpectedTensor {
        name: "layer.fp8.weight",
        shard: SHARD_A,
        span: SourceSpan::new(0, 10),
        dtype: "F8_E4M3",
        shape: &[2, 5],
    },
    ExpectedTensor {
        name: "standalone.e4m3",
        shard: SHARD_A,
        span: SourceSpan::new(14, 17),
        dtype: "F8_E4M3",
        shape: &[3],
    },
    ExpectedTensor {
        name: "unused.f32",
        shard: SHARD_C,
        span: SourceSpan::new(24, 28),
        dtype: "F32",
        shape: &[1],
    },
];

impl Fixture {
    fn new() -> Self {
        let root = TempDir::new();
        let config = br#"{"model_type":"synthetic"}"#;
        fs::write(root.path.join("config.json"), config).unwrap();

        let shard_a = write_shard(
            &root.path,
            SHARD_A,
            vec![
                TensorSource {
                    name: "layer.fp8.weight",
                    dtype: "F8_E4M3",
                    shape: &[2, 5],
                    bytes: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                TensorSource {
                    name: "layer.fp8.scale",
                    dtype: "F32",
                    shape: &[1, 1],
                    bytes: vec![0x00, 0x00, 0x80, 0x3f],
                },
                TensorSource {
                    name: "standalone.e4m3",
                    dtype: "F8_E4M3",
                    shape: &[3],
                    bytes: vec![0x11, 0x22, 0x33],
                },
            ],
        );

        let mut fp4_weight = vec![0u8; 36];
        fp4_weight[0] = 0x51;
        fp4_weight[15] = 0xac;
        fp4_weight[16] = 0x62;
        fp4_weight[17] = 0x03;
        fp4_weight[18] = 0xe4;
        fp4_weight[33] = 0x19;
        fp4_weight[34] = 0xd5;
        fp4_weight[35] = 0x04;
        let shard_b = write_shard(
            &root.path,
            SHARD_B,
            vec![
                TensorSource {
                    name: "layer.fp4.weight",
                    dtype: "I8",
                    shape: &[2, 18],
                    bytes: fp4_weight,
                },
                TensorSource {
                    name: "layer.fp4.scale",
                    dtype: "F8_E8M0",
                    shape: &[2, 2],
                    bytes: vec![0x7f, 0x80, 0x81, 0x7e],
                },
                TensorSource {
                    name: "dense.bf16",
                    dtype: "BF16",
                    shape: &[2],
                    bytes: vec![0x80, 0x3f, 0x00, 0x40],
                },
            ],
        );
        let shard_c = write_shard(
            &root.path,
            SHARD_C,
            vec![
                TensorSource {
                    name: "dense.f32",
                    dtype: "F32",
                    shape: &[2],
                    bytes: vec![0, 0, 0x80, 0x3f, 0, 0, 0, 0x40],
                },
                TensorSource {
                    name: "dense.i64",
                    dtype: "I64",
                    shape: &[2],
                    bytes: vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                },
                TensorSource {
                    name: "unused.f32",
                    dtype: "F32",
                    shape: &[1],
                    bytes: vec![0, 0, 0, 0],
                },
            ],
        );

        let mut weight_map = serde_json::Map::new();
        for (name, shard) in [
            ("layer.fp8.weight", SHARD_A),
            ("layer.fp8.scale", SHARD_A),
            ("standalone.e4m3", SHARD_A),
            ("layer.fp4.weight", SHARD_B),
            ("layer.fp4.scale", SHARD_B),
            ("dense.bf16", SHARD_B),
            ("dense.f32", SHARD_C),
            ("dense.i64", SHARD_C),
            ("unused.f32", SHARD_C),
        ] {
            weight_map.insert(name.to_string(), Value::String(shard.to_string()));
        }
        let index = serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap();
        fs::write(root.path.join("model.safetensors.index.json"), &index).unwrap();

        let manifests = vec![
            shard_a.manifest.clone(),
            shard_b.manifest.clone(),
            shard_c.manifest.clone(),
        ];
        for shard in &manifests {
            assert!(shard.file_length <= 4096);
        }
        let mut spans = BTreeMap::new();
        spans.extend(shard_a.spans);
        spans.extend(shard_b.spans);
        spans.extend(shard_c.spans);
        Self {
            root,
            manifest: PackedArtifactManifest {
                repository: "local/mixed-synthetic".to_string(),
                revision: "0123456789abcdef".to_string(),
                config_length: config.len(),
                config_sha256: sha256_digest(config),
                index_sha256: sha256_digest(&index),
                shards: manifests,
            },
            limits: PackedSafetensorsLimits {
                config_bytes: 4096,
                index_bytes: 4096,
                header_bytes_per_shard: 4096,
                shard_count: 3,
                tensor_entries: 9,
                selected_source_bytes: 85,
                packed_source_bytes: 54,
            },
            data_starts: BTreeMap::from([
                (SHARD_A.to_string(), shard_a.data_start),
                (SHARD_B.to_string(), shard_b.data_start),
                (SHARD_C.to_string(), shard_c.data_start),
            ]),
            spans,
        }
    }

    fn authenticate(&self) -> AuthenticatedSafetensorsHandleSet {
        AuthenticatedSafetensorsHandleSet::authenticate(
            &self.root.path,
            self.manifest.clone(),
            self.limits,
        )
        .unwrap()
    }

    fn refresh_manifest(&mut self, filename: &str) {
        let replacement = refresh_shard_manifest(&self.root.path, filename);
        *self
            .manifest
            .shards
            .iter_mut()
            .find(|row| row.filename == filename)
            .unwrap() = replacement;
    }
}

struct SmallFixture {
    root: TempDir,
    manifest: PackedArtifactManifest,
    limits: PackedSafetensorsLimits,
}

impl SmallFixture {
    fn new(filename: &'static str, tensors: Vec<TensorSource>) -> Self {
        let root = TempDir::new();
        let config = br#"{}"#;
        fs::write(root.path.join("config.json"), config).unwrap();
        let shard = write_shard(&root.path, filename, tensors);
        let weight_map = shard
            .spans
            .keys()
            .map(|name| (name.clone(), Value::String(filename.to_string())))
            .collect::<serde_json::Map<_, _>>();
        let index = serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap();
        fs::write(root.path.join("model.safetensors.index.json"), &index).unwrap();
        let tensor_entries = shard.spans.len();
        Self {
            root,
            manifest: PackedArtifactManifest {
                repository: "local/mixed-small".to_string(),
                revision: "small-revision".to_string(),
                config_length: config.len(),
                config_sha256: sha256_digest(config),
                index_sha256: sha256_digest(&index),
                shards: vec![shard.manifest],
            },
            limits: PackedSafetensorsLimits {
                config_bytes: 4096,
                index_bytes: 4096,
                header_bytes_per_shard: 4096,
                shard_count: 1,
                tensor_entries,
                selected_source_bytes: 4096,
                packed_source_bytes: 1,
            },
        }
    }

    fn authenticate(&self) -> AuthenticatedSafetensorsHandleSet {
        AuthenticatedSafetensorsHandleSet::authenticate(
            &self.root.path,
            self.manifest.clone(),
            self.limits,
        )
        .unwrap()
    }
}

#[derive(Clone, Copy, Debug)]
enum StandaloneMutation {
    Name,
    Shard,
    Span,
    Dtype,
    Shape,
}

fn standalone_mutation_fixture(mutation: StandaloneMutation) -> SmallFixture {
    let filename = if matches!(mutation, StandaloneMutation::Shard) {
        SHARD_B
    } else {
        SHARD_A
    };
    let mut tensors = Vec::new();
    if matches!(mutation, StandaloneMutation::Span) {
        tensors.push(TensorSource {
            name: "padding",
            dtype: "U8",
            shape: &[1],
            bytes: vec![0],
        });
    }
    tensors.push(TensorSource {
        name: if matches!(mutation, StandaloneMutation::Name) {
            "standalone.changed"
        } else {
            "standalone"
        },
        dtype: if matches!(mutation, StandaloneMutation::Dtype) {
            "U8"
        } else {
            "F8_E4M3"
        },
        shape: if matches!(mutation, StandaloneMutation::Shape) {
            &[1, 3]
        } else {
            &[3]
        },
        bytes: vec![0x11, 0x22, 0x33],
    });
    SmallFixture::new(filename, tensors)
}

fn decisions(
    inventory: AuthenticatedInventory<'_>,
    unused: TensorDisposition,
) -> Result<Vec<InventoryDecision>, ClassifierError> {
    inventory
        .rows()
        .map(|row| {
            let expected = EXPECTED_TENSORS
                .iter()
                .find(|expected| expected.name == row.name())
                .ok_or(ClassifierError::UnexpectedInventory)?;
            if row.shard() != expected.shard
                || row.span() != expected.span
                || row.dtype() != expected.dtype
                || row.shape() != expected.shape
            {
                return Err(ClassifierError::UnexpectedMetadata {
                    tensor: row.name().to_string(),
                });
            }
            let disposition = match row.name() {
                "layer.fp8.weight" => TensorDisposition::PackedWeight {
                    linear_id: "fp8".to_string(),
                    format: WeightFormat::E4m3Block128 {
                        scale: ScaleEncoding::F32,
                    },
                    logical_shape: [2, 5],
                },
                "layer.fp8.scale" => TensorDisposition::PackedScale {
                    linear_id: "fp8".to_string(),
                },
                "layer.fp4.weight" => TensorDisposition::PackedWeight {
                    linear_id: "fp4".to_string(),
                    format: WeightFormat::E2m1Row32,
                    logical_shape: [2, 35],
                },
                "layer.fp4.scale" => TensorDisposition::PackedScale {
                    linear_id: "fp4".to_string(),
                },
                "standalone.e4m3" => TensorDisposition::StandaloneE4m3,
                "dense.bf16" => TensorDisposition::DenseBf16,
                "dense.f32" => TensorDisposition::DenseF32,
                "dense.i64" => TensorDisposition::DenseI64,
                "unused.f32" => unused.clone(),
                _ => return Err(ClassifierError::UnexpectedInventory),
            };
            Ok(InventoryDecision::new(row.key(), disposition))
        })
        .collect()
}

fn exact<'a>(
    result: &'a poot_load::packed_safetensors::MixedLoadResult,
    name: &str,
) -> &'a poot_load::packed_safetensors::ExactSourceMetadata {
    result.exact_metadata.get(name).unwrap()
}

/// This linear's packed owner (`WeightStore`'s `WeightEntry::Packed`, card 540a) and its
/// checkpoint provenance (`MixedLoadResult::packed_metadata`, keyed the same way).
fn packed<'a>(
    result: &'a poot_load::packed_safetensors::MixedLoadResult,
    linear_id: &str,
) -> (
    &'a Arc<PackedPayload>,
    &'a poot_load::packed_safetensors::PackedSourceMetadata,
) {
    let owner = match result.store.get(linear_id).unwrap() {
        WeightEntry::Packed(owner) => owner,
        WeightEntry::Dense(_) => panic!("{linear_id} is a dense entry, not packed"),
    };
    (owner, result.packed_metadata.get(linear_id).unwrap())
}

fn assert_inventory(
    result: &poot_load::packed_safetensors::MixedLoadResult,
    unused: &TensorDisposition,
) {
    assert_eq!(result.inventory.len(), EXPECTED_TENSORS.len());
    for (row, expected) in result.inventory.iter().zip(&EXPECTED_TENSORS) {
        assert_eq!(row.descriptor.name(), expected.name);
        assert_eq!(row.descriptor.shard(), expected.shard);
        assert_eq!(row.descriptor.span(), expected.span);
        assert_eq!(row.descriptor.dtype(), expected.dtype);
        assert_eq!(row.descriptor.shape(), expected.shape);
        let disposition = match expected.name {
            "layer.fp8.weight" => TensorDisposition::PackedWeight {
                linear_id: "fp8".to_string(),
                format: WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::F32,
                },
                logical_shape: [2, 5],
            },
            "layer.fp8.scale" => TensorDisposition::PackedScale {
                linear_id: "fp8".to_string(),
            },
            "layer.fp4.weight" => TensorDisposition::PackedWeight {
                linear_id: "fp4".to_string(),
                format: WeightFormat::E2m1Row32,
                logical_shape: [2, 35],
            },
            "layer.fp4.scale" => TensorDisposition::PackedScale {
                linear_id: "fp4".to_string(),
            },
            "standalone.e4m3" => TensorDisposition::StandaloneE4m3,
            "dense.bf16" => TensorDisposition::DenseBf16,
            "dense.f32" => TensorDisposition::DenseF32,
            "dense.i64" => TensorDisposition::DenseI64,
            "unused.f32" => unused.clone(),
            _ => unreachable!(),
        };
        assert_eq!(row.disposition, disposition);
    }
}

#[test]
fn mixed_snapshot_preserves_exact_sources() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let result = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();

    assert_inventory(&result, &TensorDisposition::Deferred);
    assert_eq!(result.packed_metadata.len(), 2);
    assert_eq!(result.exact_metadata.len(), 4);
    assert_eq!(packed_cache.len(), 2);
    assert_eq!(exact_cache.len(), 4);
    assert_eq!(result.artifact.repository(), "local/mixed-synthetic");
    assert_eq!(result.artifact.revision(), "0123456789abcdef");
    assert_eq!(result.artifact.config_length(), CONFIG_BYTES);
    assert_eq!(
        result.artifact.config_sha256(),
        fixture.manifest.config_sha256
    );
    assert_eq!(result.artifact.index_length(), INDEX_BYTES);
    assert_eq!(
        result.artifact.index_sha256(),
        fixture.manifest.index_sha256
    );
    assert_eq!(result.artifact.shards().len(), 3);
    assert_eq!(result.artifact.shards(), fixture.manifest.shards.as_slice());
    assert_eq!(
        result
            .artifact
            .shards()
            .iter()
            .map(|shard| shard.header_length)
            .collect::<Vec<_>>(),
        vec![HEADER_A_BYTES, HEADER_B_BYTES, HEADER_C_BYTES]
    );

    let (fp8_owner, fp8) = packed(&result, "fp8");
    assert_eq!(fp8_owner.weight().shape(), [2, 5]);
    assert_eq!(fp8.weight_name, "layer.fp8.weight");
    assert_eq!(fp8.scale_name, "layer.fp8.scale");
    assert_eq!(fp8.shard, SHARD_A);
    assert_eq!(fp8.weight_span, SourceSpan::new(0, 10));
    assert_eq!(fp8.scale_span, SourceSpan::new(10, 14));
    assert_eq!(
        fp8_owner.bytes(SourceRole::Planar(OperandRole::Codes)),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_eq!(
        fp8_owner.bytes(SourceRole::Planar(OperandRole::Scale)),
        &[0, 0, 0x80, 0x3f]
    );
    let (fp4_owner, fp4) = packed(&result, "fp4");
    assert_eq!(fp4_owner.weight().shape(), [2, 35]);
    assert_eq!(
        fp4_owner
            .weight()
            .source_shape(SourceRole::Planar(OperandRole::Codes)),
        [2, 18]
    );
    assert_eq!(fp4.weight_name, "layer.fp4.weight");
    assert_eq!(fp4.scale_name, "layer.fp4.scale");
    assert_eq!(fp4.shard, SHARD_B);
    assert_eq!(fp4.weight_span, SourceSpan::new(0, 36));
    assert_eq!(fp4.scale_span, SourceSpan::new(36, 40));
    assert_eq!(
        fp4_owner.bytes(SourceRole::Planar(OperandRole::Codes))[17],
        0x03
    );
    assert_eq!(
        fp4_owner.bytes(SourceRole::Planar(OperandRole::Codes))[35],
        0x04
    );

    assert_eq!(
        exact(&result, "standalone.e4m3").owner.bytes(),
        &[0x11, 0x22, 0x33]
    );
    assert_eq!(
        exact(&result, "standalone.e4m3").kind,
        ExactSourceKind::E4m3
    );
    assert_eq!(
        exact(&result, "dense.bf16").owner.bytes(),
        &[0x80, 0x3f, 0, 0x40]
    );
    assert_eq!(exact(&result, "dense.bf16").kind, ExactSourceKind::Bf16);
    assert_eq!(
        exact(&result, "dense.f32").owner.bytes(),
        &[0, 0, 0x80, 0x3f, 0, 0, 0, 0x40]
    );
    assert_eq!(exact(&result, "dense.f32").kind, ExactSourceKind::F32);
    assert_eq!(
        exact(&result, "dense.i64").owner.bytes(),
        &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(exact(&result, "dense.i64").kind, ExactSourceKind::I64);
    for (name, meta) in &result.exact_metadata {
        assert_eq!(meta.owner.artifact(), &result.artifact);
        assert_eq!(meta.owner.descriptor().name(), name.as_str());
        assert_eq!(meta.owner.descriptor().shard(), meta.shard);
        assert_eq!(meta.owner.descriptor().span(), meta.span);
        assert_eq!(meta.owner.kind(), meta.kind);
    }

    let report = result.report;
    let source = [54, 3, 4, 8, 16, 4];
    let selected = [54, 3, 4, 8, 16, 0];
    let in_flight = [40, 3, 4, 8, 16, 0];
    for (index, category) in MixedSourceCategory::ALL.into_iter().enumerate() {
        assert_eq!(report.source_bytes.get(category), source[index]);
        assert_eq!(report.selected_source_bytes.get(category), selected[index]);
        assert_eq!(
            report.selected_range_read_bytes.get(category),
            selected[index]
        );
        assert_eq!(report.new_owner_bytes.get(category), selected[index]);
        assert_eq!(report.reused_owner_bytes.get(category), 0);
        assert_eq!(report.aggregate_owner_bytes.get(category), selected[index]);
        assert_eq!(
            report.peak_in_flight_read_bytes.get(category),
            in_flight[index]
        );
        assert_eq!(
            report.peak_unpublished_owner_bytes.get(category),
            selected[index]
        );
    }
    assert_eq!(report.source_bytes.aggregate(), 89);
    assert_eq!(report.selected_source_bytes.aggregate(), 85);
    assert_eq!(report.selected_range_read_bytes.aggregate(), 85);
    assert_eq!(report.new_owner_bytes.aggregate(), 85);
    assert_eq!(report.reused_owner_bytes.aggregate(), 0);
    assert_eq!(report.aggregate_owner_bytes.aggregate(), 85);
    assert_eq!(report.peak_in_flight_read_bytes.aggregate(), 40);
    assert_eq!(report.peak_unpublished_owner_bytes.aggregate(), 85);
    assert_eq!(report.packed_payload_builds, 2);
    assert_eq!(report.exact_source_builds.get(ExactSourceKind::E4m3), 1);
    assert_eq!(report.exact_source_builds.get(ExactSourceKind::Bf16), 1);
    assert_eq!(report.exact_source_builds.get(ExactSourceKind::F32), 1);
    assert_eq!(report.exact_source_builds.get(ExactSourceKind::I64), 1);
    assert_eq!(report.exact_source_builds.total(), 4);
    assert_eq!(report.packed_weight_source_bytes, 46);
    assert_eq!(report.packed_scale_source_bytes, 8);
    assert_eq!(report.packed_source_padding_bits, 8);
    assert_eq!(report.forbidden_f32_weight_bytes, 320);
    assert_eq!(report.config_bytes, CONFIG_BYTES);
    assert_eq!(report.index_bytes, INDEX_BYTES);
    assert_eq!(report.header_bytes_total, HEADER_TOTAL_BYTES);
    assert_eq!(report.header_bytes_max, HEADER_A_BYTES);
    assert_eq!(report.metadata_bytes, 1119);
    assert_eq!(report.initial_hash_io_bytes, CONFIG_INDEX_HASH_IO_BYTES);
    assert_eq!(report.final_hash_io_bytes, SHARD_HASH_IO_BYTES);
    assert_eq!(report.hash_buffer_bytes, 4096);
    // Card 540a: stored_bytes is WeightStore::total_stored_bytes() over this
    // snapshot's own store (dense 3+4+8+16=31 plus packed weight+scale 46+8=54 = 85), read off
    // MixedLoadResult::store itself - the load output, not a second hand-kept count.
    assert_eq!(report.stored_bytes, 85);
    assert_eq!(result.store.total_stored_bytes(), 85);
}

#[test]
fn mixed_snapshot_reuses_authoritative_owners() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let cold = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();
    let warm = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();

    assert_eq!(cold.packed_metadata.len(), 2);
    assert_eq!(warm.packed_metadata.len(), 2);
    assert_eq!(packed_cache.len(), 2);
    assert_eq!(cold.exact_metadata.len(), 4);
    assert_eq!(warm.exact_metadata.len(), 4);
    assert_eq!(exact_cache.len(), 4);
    for linear_id in cold.packed_metadata.keys() {
        let (cold_owner, _) = packed(&cold, linear_id);
        let (warm_owner, _) = packed(&warm, linear_id);
        assert!(Arc::ptr_eq(cold_owner, warm_owner));
    }
    for name in cold.exact_metadata.keys() {
        let cold_row = exact(&cold, name);
        let warm_row = exact(&warm, name);
        assert!(Arc::ptr_eq(&cold_row.owner, &warm_row.owner));
    }
    for (index, category) in MixedSourceCategory::ALL.into_iter().enumerate() {
        let expected = [54, 3, 4, 8, 16, 0][index];
        assert_eq!(warm.report.new_owner_bytes.get(category), 0);
        assert_eq!(warm.report.reused_owner_bytes.get(category), expected);
        assert_eq!(warm.report.aggregate_owner_bytes.get(category), expected);
        assert_eq!(warm.report.selected_range_read_bytes.get(category), 0);
        assert_eq!(warm.report.peak_unpublished_owner_bytes.get(category), 0);
    }
    assert_eq!(warm.report.new_owner_bytes.aggregate(), 0);
    assert_eq!(warm.report.reused_owner_bytes.aggregate(), 85);
    assert_eq!(warm.report.selected_range_read_bytes.aggregate(), 0);
    assert_eq!(warm.report.peak_in_flight_read_bytes.aggregate(), 0);
    assert_eq!(warm.report.peak_unpublished_owner_bytes.aggregate(), 0);
    assert_eq!(warm.report.packed_payload_builds, 0);
    assert_eq!(warm.report.exact_source_builds.total(), 0);
}

#[test]
fn mixed_snapshot_warm_unpublished_is_independent() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let cold = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();
    let warm = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();

    assert_eq!(cold.report.peak_unpublished_owner_bytes.aggregate(), 85);
    assert_eq!(warm.report.peak_unpublished_owner_bytes.aggregate(), 0);
    assert_ne!(cold.report.packed_source_padding_bits, 0);
    assert_ne!(cold.report.packed_payload_builds, 0);
}

#[test]
fn authenticated_inventory_drives_classification() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let result = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            assert_eq!(inventory.len(), EXPECTED_TENSORS.len());
            assert!(!inventory.is_empty());
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();

    assert_inventory(&result, &TensorDisposition::Deferred);
    for (name, meta) in &result.exact_metadata {
        assert_eq!(meta.owner.descriptor().name(), name.as_str());
    }
}

#[test]
fn inventory_decisions_cannot_supply_physical_source_fields() {
    let fixture = Fixture::new();
    let authenticated = fixture.authenticate();
    let key = authenticated.inventory().rows().next().unwrap().key();
    let decision = INVENTORY_DECISION_NEW(key.clone(), TensorDisposition::Deferred);
    let (returned_key, returned_disposition) = inventory_decision_parts(decision);
    assert_eq!(returned_key, key);
    assert_eq!(returned_disposition, TensorDisposition::Deferred);
}

#[test]
fn mixed_snapshot_preserves_classifier_error_type() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = authenticated
        .load_mixed(
            &mut packed_cache,
            &mut exact_cache,
            |_| -> Result<Vec<InventoryDecision>, ClassifierError> {
                Err(ClassifierError::Sentinel { code: 37 })
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        MixedLoadError::Classifier(ClassifierError::Sentinel { code: 37 })
    ));
    assert!(packed_cache.is_empty());
    assert!(exact_cache.is_empty());
}

#[test]
fn mixed_snapshot_partition_is_complete_and_disjoint() {
    let fixture = Fixture::new();
    let foreign_fixture = Fixture::new();
    let foreign = foreign_fixture.authenticate();
    let foreign_key = foreign.inventory().rows().next().unwrap().key();

    for case in 0..7 {
        let mut authenticated = fixture.authenticate();
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let error = authenticated
            .load_mixed(
                &mut packed_cache,
                &mut exact_cache,
                |inventory| -> Result<Vec<InventoryDecision>, ClassifierError> {
                    let mut rows = decisions(inventory, TensorDisposition::Deferred)?;
                    match case {
                        0 => {
                            rows.pop();
                        }
                        1 => rows.push(rows[0].clone()),
                        2 => rows.push(InventoryDecision::new(
                            foreign_key.clone(),
                            TensorDisposition::Deferred,
                        )),
                        3 => {
                            rows.iter_mut()
                                .find(|row| {
                                    matches!(
                                        &row.disposition,
                                        TensorDisposition::PackedWeight { linear_id, .. }
                                            if linear_id == "fp8"
                                    )
                                })
                                .unwrap()
                                .disposition = TensorDisposition::StandaloneE4m3;
                        }
                        4 => {
                            rows.iter_mut()
                                .find(|row| {
                                    matches!(
                                        &row.disposition,
                                        TensorDisposition::PackedScale { linear_id }
                                            if linear_id == "fp4"
                                    )
                                })
                                .unwrap()
                                .disposition = TensorDisposition::Deferred;
                        }
                        5 => {
                            let row = rows
                                .iter_mut()
                                .find(|row| {
                                    matches!(
                                        &row.disposition,
                                        TensorDisposition::PackedScale { linear_id }
                                            if linear_id == "fp8"
                                    )
                                })
                                .unwrap();
                            if let TensorDisposition::PackedScale { linear_id } =
                                &mut row.disposition
                            {
                                *linear_id = "changed".to_string();
                            }
                        }
                        6 => {
                            rows.iter_mut()
                                .find(|row| matches!(&row.disposition, TensorDisposition::DenseF32))
                                .unwrap()
                                .disposition = TensorDisposition::DenseI64;
                        }
                        _ => unreachable!(),
                    }
                    Ok(rows)
                },
            )
            .unwrap_err();
        match case {
            0 => assert!(matches!(
                error,
                MixedLoadError::Assembly(PackedSafetensorsError::MissingInventoryDecision { .. })
            )),
            1 => assert!(matches!(
                error,
                MixedLoadError::Assembly(PackedSafetensorsError::DuplicateInventoryKey { .. })
            )),
            2 => assert!(matches!(
                error,
                MixedLoadError::Assembly(PackedSafetensorsError::ForeignInventoryKey { .. })
            )),
            3..=5 => assert!(matches!(
                error,
                MixedLoadError::Assembly(PackedSafetensorsError::MissingPackedComponent { .. })
            )),
            6 => assert!(matches!(
                error,
                MixedLoadError::Assembly(PackedSafetensorsError::DispositionDtypeMismatch { .. })
            )),
            _ => unreachable!(),
        }
        assert!(packed_cache.is_empty());
        assert!(exact_cache.is_empty());
    }
}

#[test]
fn mixed_snapshot_rejects_foreign_inventory_keys() {
    let fixture = Fixture::new();
    let foreign_fixture = Fixture::new();
    let foreign = foreign_fixture.authenticate();
    let foreign_key = foreign.inventory().rows().next().unwrap().key();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = authenticated
        .load_mixed(
            &mut packed_cache,
            &mut exact_cache,
            |inventory| -> Result<Vec<InventoryDecision>, ClassifierError> {
                let mut rows = decisions(inventory, TensorDisposition::Deferred)?;
                rows[0].key = foreign_key;
                Ok(rows)
            },
        )
        .unwrap_err();

    assert!(matches!(
        error,
        MixedLoadError::Assembly(PackedSafetensorsError::ForeignInventoryKey { .. })
    ));
    assert!(packed_cache.is_empty());
    assert!(exact_cache.is_empty());
}

#[test]
fn mixed_snapshot_rejects_duplicate_empty_selected_spans() {
    let fixture = SmallFixture::new(
        SHARD_A,
        vec![
            TensorSource {
                name: "empty.a",
                dtype: "F32",
                shape: &[0],
                bytes: Vec::new(),
            },
            TensorSource {
                name: "empty.b",
                dtype: "F32",
                shape: &[0],
                bytes: Vec::new(),
            },
        ],
    );
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            Ok::<_, ClassifierError>(
                inventory
                    .rows()
                    .map(|row| InventoryDecision::new(row.key(), TensorDisposition::DenseF32))
                    .collect(),
            )
        })
        .unwrap_err();

    assert!(matches!(
        error,
        MixedLoadError::Assembly(PackedSafetensorsError::ReusedSourceSpan {
            span: SourceSpan { start: 0, end: 0 },
            ..
        })
    ));
    assert!(packed_cache.is_empty());
    assert!(exact_cache.is_empty());
}

#[test]
fn mixed_snapshot_rejects_standalone_metadata_mutations() {
    for mutation in [
        StandaloneMutation::Name,
        StandaloneMutation::Shard,
        StandaloneMutation::Span,
        StandaloneMutation::Dtype,
        StandaloneMutation::Shape,
    ] {
        let fixture = standalone_mutation_fixture(mutation);
        let mut authenticated = fixture.authenticate();
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let error = authenticated
            .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
                inventory
                    .rows()
                    .map(|row| {
                        if row.name() == "padding" {
                            return Ok(InventoryDecision::new(
                                row.key(),
                                TensorDisposition::Deferred,
                            ));
                        }
                        if row.name() != "standalone" {
                            return Err(ClassifierError::UnexpectedInventory);
                        }
                        if row.shard() != SHARD_A
                            || row.span() != SourceSpan::new(0, 3)
                            || row.dtype() != "F8_E4M3"
                            || row.shape() != [3].as_slice()
                        {
                            return Err(ClassifierError::UnexpectedMetadata {
                                tensor: row.name().to_string(),
                            });
                        }
                        Ok(InventoryDecision::new(
                            row.key(),
                            TensorDisposition::StandaloneE4m3,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap_err();

        assert!(
            matches!(error, MixedLoadError::Classifier(_)),
            "{mutation:?} escaped classifier validation"
        );
        assert!(packed_cache.is_empty());
        assert!(exact_cache.is_empty());
    }
}

#[test]
fn mixed_snapshot_ragged_logical_shape_is_explicit() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = authenticated
        .load_mixed(
            &mut packed_cache,
            &mut exact_cache,
            |inventory| -> Result<Vec<InventoryDecision>, ClassifierError> {
                let mut rows = decisions(inventory, TensorDisposition::Deferred)?;
                let row = rows
                    .iter_mut()
                    .find(|row| {
                        matches!(
                            &row.disposition,
                            TensorDisposition::PackedWeight { linear_id, .. } if linear_id == "fp4"
                        )
                    })
                    .unwrap();
                if let TensorDisposition::PackedWeight { logical_shape, .. } = &mut row.disposition
                {
                    *logical_shape = [2, 37];
                }
                Ok(rows)
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        MixedLoadError::Assembly(PackedSafetensorsError::SelectionMismatch { .. })
    ));
    assert!(packed_cache.is_empty());
    assert!(exact_cache.is_empty());
}

#[test]
fn mixed_snapshot_source_cache_separates_artifacts() {
    let first = Fixture::new();
    let mut first_authenticated = first.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let first_result = first_authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();

    let mut second = Fixture::new();
    let offset = second.data_starts[SHARD_A] + second.spans["standalone.e4m3"].start;
    write_at(&second.root.path.join(SHARD_A), offset, 0x44);
    second.refresh_manifest(SHARD_A);
    let mut second_authenticated = second.authenticate();
    let mut second_packed_cache = PackedOwnerCache::new();
    let second_result = second_authenticated
        .load_mixed(&mut second_packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();

    assert_eq!(exact_cache.len(), 8);
    assert_eq!(second_result.report.exact_source_builds.total(), 4);
    for (category, bytes) in [
        (MixedSourceCategory::StandaloneE4m3, 3),
        (MixedSourceCategory::DenseBf16, 4),
        (MixedSourceCategory::DenseF32, 8),
        (MixedSourceCategory::DenseI64, 16),
    ] {
        assert_eq!(second_result.report.new_owner_bytes.get(category), bytes);
        assert_eq!(
            second_result.report.selected_range_read_bytes.get(category),
            bytes
        );
        assert_eq!(second_result.report.reused_owner_bytes.get(category), 0);
    }
    assert_eq!(
        exact(&second_result, "standalone.e4m3").owner.bytes()[0],
        0x44
    );
    for name in first_result.exact_metadata.keys() {
        let first_row = exact(&first_result, name);
        let second_row = exact(&second_result, name);
        assert!(!Arc::ptr_eq(&first_row.owner, &second_row.owner));
    }
}

#[test]
fn mixed_snapshot_deferred_and_excluded_are_metadata_only() {
    for unused in [TensorDisposition::Deferred, TensorDisposition::Excluded] {
        let fixture = Fixture::new();
        let mut authenticated = fixture.authenticate();
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let result = authenticated
            .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
                decisions(inventory, unused.clone())
            })
            .unwrap();
        let row = result
            .inventory
            .iter()
            .find(|row| row.descriptor.name() == "unused.f32")
            .unwrap();
        assert_eq!(row.disposition, unused);
        assert!(!result.exact_metadata.contains_key("unused.f32"));
        assert_eq!(
            result
                .report
                .source_bytes
                .get(MixedSourceCategory::DeferredOrExcluded),
            4
        );
        assert_eq!(
            result
                .report
                .selected_source_bytes
                .get(MixedSourceCategory::DeferredOrExcluded),
            0
        );
        assert_eq!(
            result
                .report
                .selected_range_read_bytes
                .get(MixedSourceCategory::DeferredOrExcluded),
            0
        );
        assert_eq!(
            result
                .report
                .aggregate_owner_bytes
                .get(MixedSourceCategory::DeferredOrExcluded),
            0
        );
        assert_eq!(
            result
                .report
                .selected_source_bytes
                .get(MixedSourceCategory::DenseF32),
            8
        );
        assert_eq!(result.report.selected_source_bytes.aggregate(), 85);
        assert_eq!(result.report.exact_source_builds.total(), 4);
        assert_eq!(result.report.final_hash_io_bytes, SHARD_HASH_IO_BYTES);

        let warm = authenticated
            .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
                decisions(inventory, unused.clone())
            })
            .unwrap();
        assert_eq!(warm.report.exact_source_builds.total(), 0);
        assert_eq!(warm.report.selected_range_read_bytes.aggregate(), 0);
        assert_eq!(warm.report.new_owner_bytes.aggregate(), 0);
    }
}

#[test]
fn mixed_snapshot_final_verification_is_atomic() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let cold = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();
    let packed_len = packed_cache.len();
    let exact_len = exact_cache.len();
    let offset = fixture.data_starts[SHARD_A] + fixture.spans["standalone.e4m3"].start;
    write_at(&fixture.root.path.join(SHARD_A), offset, 0x77);
    let error = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap_err();
    assert!(matches!(
        error,
        MixedLoadError::Assembly(PackedSafetensorsError::ArtifactChanged { .. })
    ));
    assert_eq!(packed_cache.len(), packed_len);
    assert_eq!(exact_cache.len(), exact_len);

    write_at(&fixture.root.path.join(SHARD_A), offset, 0x11);
    let warm = authenticated
        .load_mixed(&mut packed_cache, &mut exact_cache, |inventory| {
            decisions(inventory, TensorDisposition::Deferred)
        })
        .unwrap();
    for linear_id in cold.packed_metadata.keys() {
        let (cold_owner, _) = packed(&cold, linear_id);
        let (warm_owner, _) = packed(&warm, linear_id);
        assert!(Arc::ptr_eq(cold_owner, warm_owner));
    }
    for name in cold.exact_metadata.keys() {
        assert!(Arc::ptr_eq(
            &exact(&cold, name).owner,
            &exact(&warm, name).owner
        ));
    }
}

/// Tensor names one code-sweep snapshot is written under. The loader must not care which.
struct SweepNames {
    fp8_weight: &'static str,
    fp8_scale: &'static str,
    fp4_weight: &'static str,
    fp4_scale: &'static str,
    standalone: &'static str,
}

/// The source bytes of the code-sweep snapshot: every E4M3 code the loader admits (all but the two NaN
/// encodings 0x7f and 0xff, which it refuses) as the FP8 weight, every one of the 256 byte values
/// (reversed) as the packed FP4 weight so every nibble pair is present, an FP8 scale of 2.0, and a
/// standalone E4M3 tensor of the largest code, negative zero and the most negative code.
const SWEEP_STANDALONE: [u8; 3] = [0x7e, 0x80, 0xfe];
const SWEEP_FP8_SCALE: [u8; 4] = [0, 0, 0, 0x40];
const SWEEP_FP4_SCALE: [u8; 16] = [
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
];

fn sweep_fp8_weight() -> Vec<u8> {
    (0..=255u8)
        .filter(|code| code & 0x7f != 0x7f)
        .chain([0x7e, 0xfe])
        .collect()
}

fn sweep_fp4_weight() -> Vec<u8> {
    (0..=255u8).rev().collect()
}

/// Write the code-sweep snapshot under `names` and load it as a mixed snapshot, classifying each
/// tensor from the name the caller chose for it.
fn load_code_sweep(names: &SweepNames) -> poot_load::packed_safetensors::MixedLoadResult {
    let mut fixture = SmallFixture::new(
        SHARD_A,
        vec![
            TensorSource {
                name: names.fp8_weight,
                dtype: "F8_E4M3",
                shape: &[16, 16],
                bytes: sweep_fp8_weight(),
            },
            TensorSource {
                name: names.fp8_scale,
                dtype: "F32",
                shape: &[1, 1],
                bytes: SWEEP_FP8_SCALE.to_vec(),
            },
            TensorSource {
                name: names.fp4_weight,
                dtype: "I8",
                shape: &[4, 64],
                bytes: sweep_fp4_weight(),
            },
            TensorSource {
                name: names.fp4_scale,
                dtype: "F8_E8M0",
                shape: &[4, 4],
                bytes: SWEEP_FP4_SCALE.to_vec(),
            },
            TensorSource {
                name: names.standalone,
                dtype: "F8_E4M3",
                shape: &[3],
                bytes: SWEEP_STANDALONE.to_vec(),
            },
        ],
    );
    fixture.limits.packed_source_bytes = 4096;
    let mut authenticated = fixture.authenticate();
    authenticated
        .load_mixed(
            &mut PackedOwnerCache::new(),
            &mut ExactSourceOwnerCache::new(),
            |inventory| {
                inventory
                    .rows()
                    .map(|row| {
                        let disposition = match row.name() {
                            name if name == names.fp8_weight => TensorDisposition::PackedWeight {
                                linear_id: "fp8".to_string(),
                                format: WeightFormat::E4m3Block128 {
                                    scale: ScaleEncoding::F32,
                                },
                                logical_shape: [16, 16],
                            },
                            name if name == names.fp8_scale => TensorDisposition::PackedScale {
                                linear_id: "fp8".to_string(),
                            },
                            name if name == names.fp4_weight => TensorDisposition::PackedWeight {
                                linear_id: "fp4".to_string(),
                                format: WeightFormat::E2m1Row32,
                                logical_shape: [4, 128],
                            },
                            name if name == names.fp4_scale => TensorDisposition::PackedScale {
                                linear_id: "fp4".to_string(),
                            },
                            name if name == names.standalone => TensorDisposition::StandaloneE4m3,
                            _ => return Err(ClassifierError::UnexpectedInventory),
                        };
                        Ok(InventoryDecision::new(row.key(), disposition))
                    })
                    .collect()
            },
        )
        .unwrap_or_else(|error| panic!("code-sweep load failed: {error}"))
}

const PLAIN_NAMES: SweepNames = SweepNames {
    fp8_weight: "a.weight",
    fp8_scale: "a.scale",
    fp4_weight: "b.weight",
    fp4_scale: "b.scale",
    standalone: "c",
};

/// A packed pair is admitted as its exact source bytes: the FP8 weight is 256 bytes (not the 1024 of
/// a widened f32 copy) and every byte, including negative zero and the subnormal and extreme codes
/// that a decode to f32 and back does not reproduce, is the byte the shard holds. The same holds for the packed FP4
/// weight, the scales and a standalone E4M3 tensor. Mutation observed red: the staged FP8 weight is
/// rebuilt as `f8_e4m3_to_f32(byte).to_le_bytes()` per byte before the payload is built.
#[test]
fn mixed_snapshot_admits_packed_bytes_verbatim() {
    let result = load_code_sweep(&PLAIN_NAMES);

    let (fp8_owner, _) = packed(&result, "fp8");
    assert_eq!(
        fp8_owner.bytes(SourceRole::Planar(OperandRole::Codes)),
        sweep_fp8_weight().as_slice()
    );
    assert_eq!(
        fp8_owner.bytes(SourceRole::Planar(OperandRole::Scale)),
        &SWEEP_FP8_SCALE
    );
    let (fp4_owner, _) = packed(&result, "fp4");
    assert_eq!(
        fp4_owner.bytes(SourceRole::Planar(OperandRole::Codes)),
        sweep_fp4_weight().as_slice()
    );
    assert_eq!(
        fp4_owner.bytes(SourceRole::Planar(OperandRole::Scale)),
        &SWEEP_FP4_SCALE
    );
    assert_eq!(exact(&result, "c").owner.bytes(), &SWEEP_STANDALONE);

    assert_eq!(result.report.packed_weight_source_bytes, 256 + 256);
    assert_eq!(
        result.report.aggregate_owner_bytes.aggregate(),
        256 + 4 + 256 + 16 + 3,
        "the owners hold exactly the source bytes"
    );
    assert_eq!(
        result.report.forbidden_f32_weight_bytes,
        4 * (16 * 16 + 4 * 128),
        "the f32 mirror the loader must not build is four times the logical element count"
    );
}

/// The loader reads what the caller's classifier says about a tensor, never what its name says:
/// the same snapshot written under names of other model families (and role words a loader might be
/// tempted to key on) loads to the same owners, bytes and payload accounting, and each row carries
/// back the name the caller wrote. Mutation observed red: the loader treats a tensor whose name
/// contains "qwen" as excluded.
#[test]
fn mixed_snapshot_is_model_neutral() {
    let reference = load_code_sweep(&PLAIN_NAMES);
    let namings = [
        SweepNames {
            fp8_weight: "qwen2.layers.0.mlp.gate_proj.weight",
            fp8_scale: "qwen2.layers.0.mlp.gate_proj.weight_scale",
            fp4_weight: "qwen2.layers.0.mlp.up_proj.weight",
            fp4_scale: "qwen2.layers.0.mlp.up_proj.weight_scale",
            standalone: "qwen2.embed_tokens.weight",
        },
        SweepNames {
            fp8_weight: "deepseek.blocks.3.experts.7.w1",
            fp8_scale: "deepseek.blocks.3.experts.7.w1_scale",
            fp4_weight: "deepseek.blocks.3.experts.7.w2",
            fp4_scale: "deepseek.blocks.3.experts.7.w2_scale",
            standalone: "deepseek.norm",
        },
        SweepNames {
            fp8_weight: "layer_role.expert_idx.tensor_name.weight",
            fp8_scale: "layer_role.expert_idx.tensor_name.scale",
            fp4_weight: "minimax.expert_role.weight",
            fp4_scale: "minimax.expert_role.scale",
            standalone: "glm.model_semantic.tensor_suffix",
        },
    ];
    for names in &namings {
        let result = load_code_sweep(names);
        // `BTreeMap` iterates in key (linear id) order, so both sides are already sorted the same
        // way the old `Vec<LoadedPackedLinear>` was sorted by hand.
        let got_ids: Vec<&String> = result.packed_metadata.keys().collect();
        let want_ids: Vec<&String> = reference.packed_metadata.keys().collect();
        assert_eq!(got_ids.len(), want_ids.len(), "{}", names.fp8_weight);
        for (linear_id, want_id) in got_ids.iter().zip(&want_ids) {
            assert_eq!(linear_id, want_id);
            let (got_owner, got_meta) = packed(&result, linear_id);
            let (want_owner, want_meta) = packed(&reference, linear_id);
            assert_eq!(
                got_owner.bytes(SourceRole::Planar(OperandRole::Codes)),
                want_owner.bytes(SourceRole::Planar(OperandRole::Codes))
            );
            assert_eq!(
                got_owner.bytes(SourceRole::Planar(OperandRole::Scale)),
                want_owner.bytes(SourceRole::Planar(OperandRole::Scale))
            );
            assert_eq!(got_meta.weight_span, want_meta.weight_span);
            assert_eq!(got_meta.scale_span, want_meta.scale_span);
        }
        let (_, fp8) = packed(&result, "fp8");
        assert_eq!(fp8.weight_name, names.fp8_weight);
        assert_eq!(fp8.scale_name, names.fp8_scale);
        assert_eq!(
            exact(&result, names.standalone).owner.bytes(),
            &SWEEP_STANDALONE
        );

        let (got, want) = (result.report, reference.report);
        assert_eq!(got.packed_payload_builds, want.packed_payload_builds);
        assert_eq!(
            got.packed_weight_source_bytes,
            want.packed_weight_source_bytes
        );
        assert_eq!(
            got.packed_scale_source_bytes,
            want.packed_scale_source_bytes
        );
        assert_eq!(got.selected_source_bytes, want.selected_source_bytes);
        assert_eq!(got.new_owner_bytes, want.new_owner_bytes);
        assert_eq!(got.aggregate_owner_bytes, want.aggregate_owner_bytes);
    }
}
