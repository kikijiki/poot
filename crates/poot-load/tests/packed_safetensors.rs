use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use poot_load::packed_safetensors::{
    AuthenticatedSafetensorsHandleSet, PackedArtifactManifest, PackedBufferKind, PackedLimitKind,
    PackedOwnerCache, PackedSafetensorsError, PackedSafetensorsLimits, PackedSelectionField,
    PackedSelectionRow, PackedShardManifest, Sha256Digest, SourceSpan, sha256_digest,
};
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedWeight, PackedWeightError, SourceRole};
use serde_json::{Value, json};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

const SHARD_A: &str = "model-00001-of-00003.safetensors";
const SHARD_B: &str = "model-00002-of-00003.safetensors";
const SHARD_C: &str = "model-00003-of-00003.safetensors";

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "poot-packed-safetensors-{}-{id}",
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

type SelectionMutation = Box<dyn Fn(&mut PackedSelectionRow)>;

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
    write_shard_parts(root, filename, Value::Object(header), data, spans)
}

fn write_shard_parts(
    root: &Path,
    filename: &str,
    header: Value,
    data: Vec<u8>,
    spans: BTreeMap<String, SourceSpan>,
) -> WrittenShard {
    let header_bytes = serde_json::to_vec(&header).unwrap();
    write_raw_shard(root, filename, &header_bytes, &data, spans)
}

fn write_raw_shard(
    root: &Path,
    filename: &str,
    header_bytes: &[u8],
    data: &[u8],
    spans: BTreeMap<String, SourceSpan>,
) -> WrittenShard {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_bytes);
    bytes.extend_from_slice(data);
    fs::write(root.join(filename), &bytes).unwrap();
    WrittenShard {
        manifest: PackedShardManifest {
            filename: filename.to_string(),
            file_length: bytes.len(),
            file_sha256: sha256_digest(&bytes),
            header_length: header_bytes.len(),
            header_sha256: sha256_digest(header_bytes),
        },
        spans,
        data_start: 8 + header_bytes.len(),
    }
}

fn read_shard_parts(path: &Path) -> (Value, Vec<u8>) {
    let bytes = fs::read(path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_length;
    (
        serde_json::from_slice(&bytes[8..data_start]).unwrap(),
        bytes[data_start..].to_vec(),
    )
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

fn replace_shard_manifest(manifest: &mut PackedArtifactManifest, replacement: PackedShardManifest) {
    let row = manifest
        .shards
        .iter_mut()
        .find(|row| row.filename == replacement.filename)
        .unwrap();
    *row = replacement;
}

/// The fixture's `weight_map`: every header tensor, in its own shard.
const FIXTURE_INDEX: [(&str, &str); 6] = [
    ("layer.fp8.weight", SHARD_A),
    ("layer.fp8.scale", SHARD_A),
    ("layer.fp4.weight", SHARD_B),
    ("layer.fp4.scale", SHARD_B),
    ("unused.a", SHARD_C),
    ("unused.b", SHARD_C),
];

/// Write `model.safetensors.index.json` with `weight_map` and return its bytes.
fn write_index(root: &Path, weight_map: &[(&str, &str)]) -> Vec<u8> {
    let weight_map: serde_json::Map<String, Value> = weight_map
        .iter()
        .map(|&(name, shard)| (name.to_string(), Value::String(shard.to_string())))
        .collect();
    let index_bytes = serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap();
    fs::write(root.join("model.safetensors.index.json"), &index_bytes).unwrap();
    index_bytes
}

fn write_at(path: &Path, offset: usize, byte: u8) {
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(offset as u64)).unwrap();
    file.write_all(&[byte]).unwrap();
    file.flush().unwrap();
}

fn adjacent_k_code(bytes: &[u8], row: usize, k: usize) -> u8 {
    let byte = bytes[row * 18 + k / 2];
    if k.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

fn row32_scale(bytes: &[u8], row: usize, k: usize) -> u8 {
    bytes[row * 2 + k / 32]
}

struct Fixture {
    root: TempDir,
    manifest: PackedArtifactManifest,
    limits: PackedSafetensorsLimits,
    selections: Vec<PackedSelectionRow>,
    data_starts: BTreeMap<String, usize>,
}

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
            ],
        );
        let shard_c = write_shard(
            &root.path,
            SHARD_C,
            vec![
                TensorSource {
                    name: "unused.a",
                    dtype: "F32",
                    shape: &[1],
                    bytes: vec![0, 0, 0, 0],
                },
                TensorSource {
                    name: "unused.b",
                    dtype: "F32",
                    shape: &[1],
                    bytes: vec![0, 0, 0, 0],
                },
            ],
        );

        let index_bytes = write_index(&root.path, &FIXTURE_INDEX);

        let fp8_descriptor = PackedWeight::try_new(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            [2, 5],
        )
        .unwrap();
        let fp4_descriptor = PackedWeight::try_new(WeightFormat::E2m1Row32, [2, 35]).unwrap();
        let selections = vec![
            PackedSelectionRow {
                linear_id: "fp8".to_string(),
                descriptor: fp8_descriptor,
                weight_name: "layer.fp8.weight".to_string(),
                scale_name: "layer.fp8.scale".to_string(),
                shard: SHARD_A.to_string(),
                weight_span: shard_a.spans["layer.fp8.weight"],
                scale_span: shard_a.spans["layer.fp8.scale"],
                weight_dtype: "F8_E4M3".to_string(),
                scale_dtype: "F32".to_string(),
                weight_shape: [2, 5],
                scale_shape: [1, 1],
            },
            PackedSelectionRow {
                linear_id: "fp4".to_string(),
                descriptor: fp4_descriptor,
                weight_name: "layer.fp4.weight".to_string(),
                scale_name: "layer.fp4.scale".to_string(),
                shard: SHARD_B.to_string(),
                weight_span: shard_b.spans["layer.fp4.weight"],
                scale_span: shard_b.spans["layer.fp4.scale"],
                weight_dtype: "I8".to_string(),
                scale_dtype: "F8_E8M0".to_string(),
                weight_shape: [2, 18],
                scale_shape: [2, 2],
            },
        ];
        let manifest = PackedArtifactManifest {
            repository: "local/synthetic".to_string(),
            revision: "0123456789abcdef".to_string(),
            config_length: config.len(),
            config_sha256: sha256_digest(config),
            index_sha256: sha256_digest(&index_bytes),
            shards: vec![
                shard_a.manifest.clone(),
                shard_b.manifest.clone(),
                shard_c.manifest.clone(),
            ],
        };
        let data_starts = BTreeMap::from([
            (SHARD_A.to_string(), shard_a.data_start),
            (SHARD_B.to_string(), shard_b.data_start),
            (SHARD_C.to_string(), shard_c.data_start),
        ]);
        for shard in &manifest.shards {
            assert!(shard.file_length <= 4096);
        }
        assert!(config.len() <= 4096);
        assert!(index_bytes.len() <= 4096);
        Self {
            root,
            manifest,
            limits: PackedSafetensorsLimits {
                config_bytes: 4096,
                index_bytes: 4096,
                header_bytes_per_shard: 4096,
                shard_count: 3,
                tensor_entries: 6,
                selected_source_bytes: 54,
                packed_source_bytes: 54,
            },
            selections,
            data_starts,
        }
    }

    fn authenticate(&self) -> AuthenticatedSafetensorsHandleSet {
        AuthenticatedSafetensorsHandleSet::authenticate(
            &self.root.path,
            self.manifest.clone(),
            self.limits,
        )
        .unwrap_or_else(|error| panic!("fixture authentication failed: {error}"))
    }

    fn authentication_error(
        &self,
        manifest: PackedArtifactManifest,
        limits: PackedSafetensorsLimits,
    ) -> PackedSafetensorsError {
        match AuthenticatedSafetensorsHandleSet::authenticate(&self.root.path, manifest, limits) {
            Ok(_) => panic!("authentication unexpectedly succeeded"),
            Err(error) => error,
        }
    }

    /// Bytes admitted once at authentication (card 544): `config.json` plus the shard index.
    /// Every shard's own full content admits once, later, at publication - `shard_file_bytes`.
    fn config_index_bytes(&self) -> usize {
        self.manifest.config_length
            + fs::metadata(self.root.path.join("model.safetensors.index.json"))
                .unwrap()
                .len() as usize
    }

    /// Every shard's full file length, summed: the one pass publication makes per shard (card
    /// 544), never the config/index bytes `config_index_bytes` already covers.
    fn shard_file_bytes(&self) -> usize {
        self.manifest
            .shards
            .iter()
            .map(|shard| shard.file_length)
            .sum::<usize>()
    }
}

#[test]
fn multishard_selection_preserves_exact_source_owners_and_order() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut cache = PackedOwnerCache::new();
    let loaded = authenticated.load(&fixture.selections, &mut cache).unwrap();

    assert_eq!(loaded.rows.len(), 2);
    assert_eq!(loaded.rows[0].linear_id, "fp8");
    assert_eq!(loaded.rows[1].linear_id, "fp4");
    assert_eq!(
        loaded.rows[0]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes)),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_eq!(
        loaded.rows[0]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Scale)),
        &[0x00, 0x00, 0x80, 0x3f]
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[0],
        0x51
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[15],
        0xac
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[16],
        0x62
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[17],
        0x03
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[18],
        0xe4
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[33],
        0x19
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[34],
        0xd5
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Codes))[35],
        0x04
    );
    assert_eq!(
        adjacent_k_code(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            0,
            31
        ),
        0x0a
    );
    assert_eq!(
        adjacent_k_code(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            0,
            32
        ),
        0x02
    );
    assert_eq!(
        adjacent_k_code(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            0,
            34
        ),
        0x03
    );
    assert_eq!(
        adjacent_k_code(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            1,
            31
        ),
        0x01
    );
    assert_eq!(
        adjacent_k_code(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            1,
            32
        ),
        0x05
    );
    assert_eq!(
        adjacent_k_code(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            1,
            34
        ),
        0x04
    );
    assert_eq!(
        loaded.rows[1]
            .owner
            .bytes(SourceRole::Planar(OperandRole::Scale)),
        &[0x7f, 0x80, 0x81, 0x7e]
    );
    assert_eq!(
        row32_scale(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Scale)),
            0,
            31
        ),
        0x7f
    );
    assert_eq!(
        row32_scale(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Scale)),
            0,
            32
        ),
        0x80
    );
    assert_eq!(
        row32_scale(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Scale)),
            1,
            31
        ),
        0x81
    );
    assert_eq!(
        row32_scale(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Scale)),
            1,
            32
        ),
        0x7e
    );

    let report = loaded.report;
    assert_eq!(report.config_bytes, 26);
    assert_eq!(
        report.header_bytes_total,
        fixture
            .manifest
            .shards
            .iter()
            .map(|shard| shard.header_length)
            .sum::<usize>()
    );
    assert_eq!(
        report.header_bytes_max,
        fixture
            .manifest
            .shards
            .iter()
            .map(|shard| shard.header_length)
            .max()
            .unwrap()
    );
    assert_eq!(report.selected_weight_source_bytes, 46);
    assert_eq!(report.selected_scale_source_bytes, 8);
    assert_eq!(report.source_padding_bits, 8);
    assert_eq!(report.packed_source_bytes, 54);
    assert_eq!(report.cold_new_owner_bytes, 54);
    assert_eq!(report.warm_reused_owner_bytes, 0);
    assert_eq!(report.selected_range_read_bytes, 54);
    assert_eq!(report.payload_builds, 2);
    assert_eq!(report.forbidden_f32_weight_bytes, 320);
    assert_eq!(
        report.initial_artifact_hash_bytes,
        fixture.config_index_bytes()
    );
    assert_eq!(report.final_verification_bytes, fixture.shard_file_bytes());
    assert_eq!(cache.len(), 2);
}

#[test]
fn artifact_header_name_span_dtype_shape_and_pair_mutations_reject() {
    let fixture = Fixture::new();

    let mut bad_artifact = fixture.manifest.clone();
    bad_artifact.config_sha256 = Sha256Digest::new([0x55; 32]);
    assert!(matches!(
        fixture.authentication_error(bad_artifact, fixture.limits),
        PackedSafetensorsError::DigestMismatch { .. }
    ));

    let mut bad_header = fixture.manifest.clone();
    bad_header.shards[0].header_sha256 = Sha256Digest::new([0xaa; 32]);
    assert!(matches!(
        fixture.authentication_error(bad_header, fixture.limits),
        PackedSafetensorsError::HeaderDigestMismatch { .. }
    ));

    let mutations: Vec<SelectionMutation> = vec![
        Box::new(|row| row.weight_name = "missing.weight".to_string()),
        Box::new(|row| row.weight_span.end -= 1),
        Box::new(|row| row.weight_dtype = "BF16".to_string()),
        Box::new(|row| row.weight_shape = [1, 10]),
        Box::new(|row| row.scale_name = "layer.fp4.scale".to_string()),
    ];
    for (index, mutate) in mutations.into_iter().enumerate() {
        let mut authenticated = fixture.authenticate();
        let mut cache = PackedOwnerCache::new();
        let mut rows = fixture.selections.clone();
        mutate(&mut rows[0]);
        let error = authenticated.load(&rows, &mut cache).unwrap_err();
        match index {
            0 => assert!(matches!(
                error,
                PackedSafetensorsError::MissingSelectionTensor { .. }
            )),
            1 => assert!(matches!(
                error,
                PackedSafetensorsError::SelectionMismatch {
                    field: PackedSelectionField::WeightSpan,
                    ..
                }
            )),
            2 => assert!(matches!(
                error,
                PackedSafetensorsError::SelectionMismatch {
                    field: PackedSelectionField::WeightDtype,
                    ..
                }
            )),
            3 => assert!(matches!(
                error,
                PackedSafetensorsError::SelectionMismatch {
                    field: PackedSelectionField::WeightShape,
                    ..
                }
            )),
            4 => assert!(matches!(error, PackedSafetensorsError::SplitPair { .. })),
            _ => unreachable!(),
        }
        assert!(cache.is_empty());
    }
}

/// Mutant H3 (mutants-m4.md): `validate_index_header_bijection -> Ok(())` admitted an artifact
/// whose index disagrees with its shard headers, and no test named the three variants. Each index
/// below keeps the shard set, so authentication reaches the bijection check, and the manifest pins
/// the rewritten index so the digest check passes.
#[test]
fn index_that_disagrees_with_the_shard_headers_rejects() {
    let fixture = Fixture::new();
    let without_unused_b: Vec<(&str, &str)> = FIXTURE_INDEX
        .into_iter()
        .filter(|&(name, _)| name != "unused.b")
        .collect();
    let unused_a_in_shard_b: Vec<(&str, &str)> = FIXTURE_INDEX
        .into_iter()
        .map(|(name, shard)| (name, if name == "unused.a" { SHARD_B } else { shard }))
        .collect();
    let with_ghost: Vec<(&str, &str)> = FIXTURE_INDEX
        .into_iter()
        .chain([("ghost.weight", SHARD_C)])
        .collect();
    let rows = [
        (
            with_ghost,
            PackedSafetensorsError::IndexTensorMissing {
                tensor: "ghost.weight".to_string(),
                shard: SHARD_C.to_string(),
            },
        ),
        (
            unused_a_in_shard_b,
            PackedSafetensorsError::IndexHeaderShardMismatch {
                tensor: "unused.a".to_string(),
                index_shard: SHARD_B.to_string(),
                header_shard: SHARD_C.to_string(),
            },
        ),
        (
            without_unused_b,
            PackedSafetensorsError::HeaderTensorMissingFromIndex {
                tensor: "unused.b".to_string(),
                shard: SHARD_C.to_string(),
            },
        ),
    ];
    for (weight_map, expected) in rows {
        let index_bytes = write_index(&fixture.root.path, &weight_map);
        let mut manifest = fixture.manifest.clone();
        manifest.index_sha256 = sha256_digest(&index_bytes);
        let error = fixture.authentication_error(manifest, fixture.limits);
        // The error has no `PartialEq` (it carries I/O errors); `Debug` names the variant and
        // every field.
        assert_eq!(format!("{error:?}"), format!("{expected:?}"));
    }

    write_index(&fixture.root.path, &FIXTURE_INDEX);
    fixture.authenticate();
}

#[test]
fn unselected_overlap_rejects_before_selection() {
    let mut fixture = Fixture::new();
    let path = fixture.root.path.join(SHARD_C);
    let (mut header, data) = read_shard_parts(&path);
    header["unused.b"]["data_offsets"] = json!([2, 6]);
    let spans = BTreeMap::from([
        ("unused.a".to_string(), SourceSpan::new(0, 4)),
        ("unused.b".to_string(), SourceSpan::new(2, 6)),
    ]);
    let rewritten = write_shard_parts(&fixture.root.path, SHARD_C, header, data, spans);
    replace_shard_manifest(&mut fixture.manifest, rewritten.manifest);

    let error = fixture.authentication_error(fixture.manifest.clone(), fixture.limits);
    assert!(matches!(
        error,
        PackedSafetensorsError::OverlappingTensorSpans { .. }
    ));

    let (mut header, data) = read_shard_parts(&path);
    header["unused.b"]["data_offsets"] = json!([4, 8]);
    let spans = BTreeMap::from([
        ("unused.a".to_string(), SourceSpan::new(0, 4)),
        ("unused.b".to_string(), SourceSpan::new(4, 8)),
    ]);
    let rewritten = write_shard_parts(&fixture.root.path, SHARD_C, header, data, spans);
    replace_shard_manifest(&mut fixture.manifest, rewritten.manifest);
    let mut authenticated = fixture.authenticate();
    let mut cache = PackedOwnerCache::new();
    assert_eq!(
        authenticated
            .load(&fixture.selections, &mut cache)
            .unwrap()
            .rows
            .len(),
        2
    );
}

#[test]
fn split_and_duplicate_pairs_reject_atomically() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut cache = PackedOwnerCache::new();

    let mut split = fixture.selections.clone();
    split[0].scale_name = "layer.fp4.scale".to_string();
    assert!(matches!(
        authenticated.load(&split, &mut cache),
        Err(PackedSafetensorsError::SplitPair { .. })
    ));

    let duplicate = vec![fixture.selections[0].clone(), fixture.selections[0].clone()];
    assert!(matches!(
        authenticated.load(&duplicate, &mut cache),
        Err(PackedSafetensorsError::DuplicateLinearId { .. })
    ));
    assert!(cache.is_empty());

    let mut reused_name = fixture.selections.clone();
    reused_name[1].linear_id = "reused-name".to_string();
    reused_name[1].weight_name = reused_name[0].weight_name.clone();
    assert!(matches!(
        authenticated.load(&reused_name, &mut cache),
        Err(PackedSafetensorsError::ReusedSourceName { .. })
    ));

    let mut reused_span = fixture.selections.clone();
    reused_span[1].linear_id = "reused-span".to_string();
    reused_span[1].shard = SHARD_A.to_string();
    reused_span[1].weight_span = reused_span[0].weight_span;
    assert!(matches!(
        authenticated.load(&reused_span, &mut cache),
        Err(PackedSafetensorsError::ReusedSourceSpan { .. })
    ));
    assert!(cache.is_empty());

    let mut duplicate_header = Fixture::new();
    let raw = br#"{"unused.a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"unused.a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let rewritten = write_raw_shard(
        &duplicate_header.root.path,
        SHARD_C,
        raw,
        &[0u8; 8],
        BTreeMap::new(),
    );
    replace_shard_manifest(&mut duplicate_header.manifest, rewritten.manifest);
    assert!(matches!(
        duplicate_header
            .authentication_error(duplicate_header.manifest.clone(), duplicate_header.limits),
        PackedSafetensorsError::DuplicateJsonKey { .. }
    ));

    let mut duplicate_across_shards = Fixture::new();
    let path = duplicate_across_shards.root.path.join(SHARD_C);
    let (mut header, data) = read_shard_parts(&path);
    let duplicate = header.as_object_mut().unwrap().remove("unused.a").unwrap();
    header
        .as_object_mut()
        .unwrap()
        .insert("layer.fp8.weight".to_string(), duplicate);
    let rewritten = write_shard_parts(
        &duplicate_across_shards.root.path,
        SHARD_C,
        header,
        data,
        BTreeMap::new(),
    );
    replace_shard_manifest(&mut duplicate_across_shards.manifest, rewritten.manifest);
    assert!(matches!(
        duplicate_across_shards.authentication_error(
            duplicate_across_shards.manifest.clone(),
            duplicate_across_shards.limits,
        ),
        PackedSafetensorsError::DuplicateTensorAcrossShards { .. }
    ));
}

#[test]
fn limits_bound_metadata_and_range_reads() {
    let fixture = Fixture::new();

    let mut wrong_config = fixture.manifest.clone();
    wrong_config.config_length += 1;
    assert!(matches!(
        fixture.authentication_error(wrong_config, fixture.limits),
        PackedSafetensorsError::ConfigLengthMismatch { .. }
    ));

    for kind in [
        PackedLimitKind::ConfigBytes,
        PackedLimitKind::IndexBytes,
        PackedLimitKind::HeaderBytesPerShard,
        PackedLimitKind::ShardCount,
        PackedLimitKind::TensorEntries,
    ] {
        let mut limits = fixture.limits;
        match kind {
            PackedLimitKind::ConfigBytes => {
                limits.config_bytes = fixture.manifest.config_length - 1
            }
            PackedLimitKind::IndexBytes => {
                limits.index_bytes =
                    fs::metadata(fixture.root.path.join("model.safetensors.index.json"))
                        .unwrap()
                        .len() as usize
                        - 1
            }
            PackedLimitKind::HeaderBytesPerShard => limits.header_bytes_per_shard = 1,
            PackedLimitKind::ShardCount => limits.shard_count = 2,
            PackedLimitKind::TensorEntries => limits.tensor_entries = 5,
            _ => unreachable!(),
        }
        let error = fixture.authentication_error(fixture.manifest.clone(), limits);
        assert!(matches!(
            error,
            PackedSafetensorsError::LimitExceeded { kind: actual, .. } if actual == kind
        ));
    }

    for kind in [
        PackedLimitKind::SelectedSourceBytes,
        PackedLimitKind::PackedSourceBytes,
    ] {
        let guarded = Fixture::new();
        let mut limits = guarded.limits;
        match kind {
            PackedLimitKind::SelectedSourceBytes => limits.selected_source_bytes = 53,
            PackedLimitKind::PackedSourceBytes => limits.packed_source_bytes = 53,
            _ => unreachable!(),
        }
        let mut authenticated = AuthenticatedSafetensorsHandleSet::authenticate(
            &guarded.root.path,
            guarded.manifest.clone(),
            limits,
        )
        .unwrap();
        OpenOptions::new()
            .write(true)
            .open(guarded.root.path.join(SHARD_A))
            .unwrap()
            .set_len(guarded.data_starts[SHARD_A] as u64)
            .unwrap();
        let mut cache = PackedOwnerCache::new();
        assert!(matches!(
            authenticated.load(&guarded.selections, &mut cache),
            Err(PackedSafetensorsError::LimitExceeded { kind: actual, .. }) if actual == kind
        ));
        assert!(cache.is_empty());
    }
}

#[test]
fn ragged_fp4_padding_is_zero_and_reversible() {
    let mut fixture = Fixture::new();
    let path = fixture.root.path.join(SHARD_B);
    let tail = fixture.data_starts[SHARD_B] + fixture.selections[1].weight_span.start + 35;
    write_at(&path, tail, 0x14);
    let refreshed = refresh_shard_manifest(&fixture.root.path, SHARD_B);
    replace_shard_manifest(&mut fixture.manifest, refreshed);
    let mut authenticated = fixture.authenticate();
    let mut cache = PackedOwnerCache::new();
    assert!(matches!(
        authenticated.load(&fixture.selections, &mut cache),
        Err(PackedSafetensorsError::PayloadBuild {
            error: PackedWeightError::NonZeroUnusedHighNibble { .. },
            ..
        })
    ));
    assert!(cache.is_empty());

    write_at(&path, tail, 0x04);
    let refreshed = refresh_shard_manifest(&fixture.root.path, SHARD_B);
    replace_shard_manifest(&mut fixture.manifest, refreshed);
    let mut authenticated = fixture.authenticate();
    let loaded = authenticated.load(&fixture.selections, &mut cache).unwrap();
    let descriptor = loaded.rows[1].descriptor;
    let codes_bits = descriptor
        .format()
        .descriptor()
        .planar_operand(OperandRole::Codes)
        .unwrap()
        .bits as usize;
    let padding_bits = descriptor.source_bytes(SourceRole::Planar(OperandRole::Codes)) * 8
        - descriptor.logical_values() * codes_bits;
    assert_eq!(padding_bits, 8);
}

#[test]
fn between_phase_mutation_rejects_before_publication() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut cache = PackedOwnerCache::new();
    let path = fixture.root.path.join(SHARD_A);
    let offset = fixture.data_starts[SHARD_A] + fixture.selections[0].weight_span.start;

    write_at(&path, offset, 0x02);
    assert!(matches!(
        authenticated.load(&fixture.selections, &mut cache),
        Err(PackedSafetensorsError::ArtifactChanged { .. })
    ));
    assert!(cache.is_empty());

    write_at(&path, offset, 0x01);
    let loaded = authenticated.load(&fixture.selections, &mut cache).unwrap();
    assert_eq!(loaded.rows.len(), 2);
}

#[test]
fn second_pair_read_or_build_failure_publishes_nothing() {
    let read_failure = Fixture::new();
    let mut authenticated = read_failure.authenticate();
    let mut cache = PackedOwnerCache::new();
    let original_shard = fs::read(read_failure.root.path.join(SHARD_B)).unwrap();
    let second_weight_end =
        read_failure.data_starts[SHARD_B] + read_failure.selections[1].weight_span.end - 1;
    OpenOptions::new()
        .write(true)
        .open(read_failure.root.path.join(SHARD_B))
        .unwrap()
        .set_len(second_weight_end as u64)
        .unwrap();
    assert!(matches!(
        authenticated.load(&read_failure.selections, &mut cache),
        Err(PackedSafetensorsError::SelectedRead {
            buffer: PackedBufferKind::Weight,
            completed_payloads: 1,
            ..
        })
    ));
    assert!(cache.is_empty());
    fs::write(read_failure.root.path.join(SHARD_B), original_shard).unwrap();
    let retry = authenticated
        .load(&read_failure.selections, &mut cache)
        .unwrap();
    assert_eq!(retry.report.payload_builds, 2);
    assert_eq!(cache.len(), 2);

    let build_failure = Fixture::new();
    let mut authenticated = build_failure.authenticate();
    let mut cache = PackedOwnerCache::new();
    let tail =
        build_failure.data_starts[SHARD_B] + build_failure.selections[1].weight_span.start + 35;
    write_at(&build_failure.root.path.join(SHARD_B), tail, 0x14);
    assert!(matches!(
        authenticated.load(&build_failure.selections, &mut cache),
        Err(PackedSafetensorsError::PayloadBuild {
            completed_payloads: 1,
            error: PackedWeightError::NonZeroUnusedHighNibble { .. },
            ..
        })
    ));
    assert!(cache.is_empty());
    write_at(&build_failure.root.path.join(SHARD_B), tail, 0x04);
    let retry = authenticated
        .load(&build_failure.selections, &mut cache)
        .unwrap();
    assert_eq!(retry.report.payload_builds, 2);
    assert_eq!(cache.len(), 2);
}

#[test]
fn warm_selection_reuses_payload_owners() {
    let fixture = Fixture::new();
    let mut authenticated = fixture.authenticate();
    let mut cache = PackedOwnerCache::new();
    let cold = authenticated.load(&fixture.selections, &mut cache).unwrap();
    let warm = authenticated.load(&fixture.selections, &mut cache).unwrap();

    assert!(Arc::ptr_eq(&cold.rows[0].owner, &warm.rows[0].owner));
    assert!(Arc::ptr_eq(&cold.rows[1].owner, &warm.rows[1].owner));
    assert_eq!(warm.report.cold_new_owner_bytes, 0);
    assert_eq!(warm.report.warm_reused_owner_bytes, 54);
    assert_eq!(warm.report.selected_range_read_bytes, 0);
    assert_eq!(warm.report.payload_builds, 0);
    assert_eq!(
        warm.report.final_verification_bytes,
        fixture.shard_file_bytes()
    );
    assert_eq!(cache.len(), 2);

    let path = fixture.root.path.join(SHARD_A);
    let offset = fixture.data_starts[SHARD_A] + fixture.selections[0].weight_span.start;
    write_at(&path, offset, 0xff);
    assert!(matches!(
        authenticated.load(&fixture.selections, &mut cache),
        Err(PackedSafetensorsError::ArtifactChanged { .. })
    ));
    assert_eq!(cache.len(), 2);
}

/// Mutant M9 (mutants-m4.md): `update_identity_usize -> ()` drops every length prefix from the
/// artifact cache identity, so the labels `("local/synthetic0", "123456789abcdef")` and
/// `("local/synthetic", "0123456789abcdef")` hash the same bytes. Two such artifacts sharing one
/// owner cache must stay apart: the second loads cold. Loading the first label again reuses its
/// owners, so the cache is shared and keyed by the identity.
#[test]
fn artifact_cache_identity_is_prefix_free() {
    let fixture = Fixture::new();
    let labelled = |repository: &str, revision: &str| {
        let mut manifest = fixture.manifest.clone();
        manifest.repository = repository.to_string();
        manifest.revision = revision.to_string();
        AuthenticatedSafetensorsHandleSet::authenticate(
            &fixture.root.path,
            manifest,
            fixture.limits,
        )
        .unwrap()
    };
    let mut cache = PackedOwnerCache::new();

    let first = labelled("local/synthetic0", "123456789abcdef")
        .load(&fixture.selections, &mut cache)
        .unwrap();
    assert_eq!(first.report.cold_new_owner_bytes, 54);

    let boundary_moved = labelled("local/synthetic", "0123456789abcdef")
        .load(&fixture.selections, &mut cache)
        .unwrap();
    assert_eq!(boundary_moved.report.warm_reused_owner_bytes, 0);
    assert_eq!(boundary_moved.report.cold_new_owner_bytes, 54);
    assert_eq!(cache.len(), 4);

    let same_label = labelled("local/synthetic0", "123456789abcdef")
        .load(&fixture.selections, &mut cache)
        .unwrap();
    assert_eq!(same_label.report.warm_reused_owner_bytes, 54);
    assert!(Arc::ptr_eq(&first.rows[0].owner, &same_label.rows[0].owner));
}

/// The packed loader admits the bytes the shard holds and nothing wider: every E4M3 code the loader
/// accepts (256 weight bytes for a 16x16 tile, not the 1024 of a widened f32 copy), every packed FP4
/// byte value, and the scales come back byte for byte, whatever the tensors are called. Mutation
/// observed red: the staged FP8 weight is rebuilt as `f8_e4m3_to_f32(byte).to_le_bytes()` per byte
/// before the payload is built.
#[test]
fn packed_loader_admits_every_source_code_verbatim() {
    // Every E4M3 code except the two NaN encodings the payload refuses; 254 distinct codes plus a
    // repeat of each end so the tile is a full 16x16.
    let fp8_weight: Vec<u8> = (0..=255u8)
        .filter(|code| code & 0x7f != 0x7f)
        .chain([0x7e, 0xfe])
        .collect();
    let fp4_weight: Vec<u8> = (0..=255u8).rev().collect();
    let fp8_scale = vec![0x00, 0x00, 0x00, 0x40];
    let fp4_scale: Vec<u8> = (0x70..0x80).collect();

    // [fp8 weight, fp8 scale, fp4 weight, fp4 scale]: the loader is handed these by the caller and
    // must not read a model, layer or role out of them.
    let namings: [[&'static str; 4]; 3] = [
        ["a.weight", "a.scale", "b.weight", "b.scale"],
        [
            "qwen2.layers.0.mlp.gate_proj.weight",
            "qwen2.layers.0.mlp.gate_proj.weight_scale",
            "qwen2.layers.0.mlp.up_proj.weight",
            "qwen2.layers.0.mlp.up_proj.weight_scale",
        ],
        [
            "deepseek.experts.7.w1",
            "deepseek.experts.7.w1_scale",
            "layer_role.expert_idx.tensor_name",
            "layer_role.expert_idx.tensor_scale",
        ],
    ];

    for names in namings {
        let root = TempDir::new();
        let config = br#"{}"#;
        fs::write(root.path.join("config.json"), config).unwrap();
        let shard = write_shard(
            &root.path,
            SHARD_A,
            vec![
                TensorSource {
                    name: names[0],
                    dtype: "F8_E4M3",
                    shape: &[16, 16],
                    bytes: fp8_weight.clone(),
                },
                TensorSource {
                    name: names[1],
                    dtype: "F32",
                    shape: &[1, 1],
                    bytes: fp8_scale.clone(),
                },
                TensorSource {
                    name: names[2],
                    dtype: "I8",
                    shape: &[4, 64],
                    bytes: fp4_weight.clone(),
                },
                TensorSource {
                    name: names[3],
                    dtype: "F8_E8M0",
                    shape: &[4, 4],
                    bytes: fp4_scale.clone(),
                },
            ],
        );
        let weight_map: serde_json::Map<String, Value> = names
            .iter()
            .map(|name| (name.to_string(), Value::String(SHARD_A.to_string())))
            .collect();
        let index = serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap();
        fs::write(root.path.join("model.safetensors.index.json"), &index).unwrap();

        let selection =
            |linear_id: &str,
             format: WeightFormat,
             logical: [usize; 2],
             weight: (&str, &str, [usize; 2]),
             scale: (&str, &str, [usize; 2])| PackedSelectionRow {
                linear_id: linear_id.to_string(),
                descriptor: PackedWeight::try_new(format, logical).unwrap(),
                weight_name: weight.0.to_string(),
                scale_name: scale.0.to_string(),
                shard: SHARD_A.to_string(),
                weight_span: shard.spans[weight.0],
                scale_span: shard.spans[scale.0],
                weight_dtype: weight.1.to_string(),
                scale_dtype: scale.1.to_string(),
                weight_shape: weight.2,
                scale_shape: scale.2,
            };
        let selections = vec![
            selection(
                "fp8",
                WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::F32,
                },
                [16, 16],
                (names[0], "F8_E4M3", [16, 16]),
                (names[1], "F32", [1, 1]),
            ),
            selection(
                "fp4",
                WeightFormat::E2m1Row32,
                [4, 128],
                (names[2], "I8", [4, 64]),
                (names[3], "F8_E8M0", [4, 4]),
            ),
        ];
        let manifest = PackedArtifactManifest {
            repository: "local/synthetic".to_string(),
            revision: "0123456789abcdef".to_string(),
            config_length: config.len(),
            config_sha256: sha256_digest(config),
            index_sha256: sha256_digest(&index),
            shards: vec![shard.manifest.clone()],
        };
        let limits = PackedSafetensorsLimits {
            config_bytes: 4096,
            index_bytes: 4096,
            header_bytes_per_shard: 4096,
            shard_count: 1,
            tensor_entries: 4,
            selected_source_bytes: 4096,
            packed_source_bytes: 4096,
        };
        let mut authenticated =
            AuthenticatedSafetensorsHandleSet::authenticate(&root.path, manifest, limits).unwrap();
        let loaded = authenticated
            .load(&selections, &mut PackedOwnerCache::new())
            .unwrap_or_else(|error| panic!("load under {names:?} failed: {error}"));

        assert_eq!(loaded.rows.len(), 2);
        assert_eq!(
            loaded.rows[0]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            fp8_weight.as_slice()
        );
        assert_eq!(
            loaded.rows[0]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Scale)),
            fp8_scale.as_slice()
        );
        assert_eq!(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Codes)),
            fp4_weight.as_slice()
        );
        assert_eq!(
            loaded.rows[1]
                .owner
                .bytes(SourceRole::Planar(OperandRole::Scale)),
            fp4_scale.as_slice()
        );
        assert_eq!(loaded.report.packed_source_bytes, 256 + 4 + 256 + 16);
        assert_eq!(loaded.report.cold_new_owner_bytes, 256 + 4 + 256 + 16);
    }
}
