use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::packed_safetensors::{
    PackedArtifactManifest, PackedSafetensorsLimits, PackedShardManifest, sha256_digest,
};

const TEXT_FIXTURE_WEIGHT: &str = "model.language_model.layers.3.self_attn.q_a_proj.weight";

const TEXT_FIXTURE_SCALE: &str =
    "model.language_model.layers.3.self_attn.q_a_proj.weight_scale_inv";
const TEXT_FIXTURE_MTP: &str = "model.language_model.layers.45.self_attn.o_proj.weight";

const TEXT_FIXTURE_MTP_SCALE: &str =
    "model.language_model.layers.45.self_attn.o_proj.weight_scale_inv";
const TEXT_FIXTURE_VISION: &str = "model.visual.blocks.0.attn.qkv.weight";

const TEXT_FIXTURE_BF16: &str = "model.language_model.norm.weight";

const TEXT_FIXTURE_F32: &str = "model.language_model.layers.0.mlp.gate.e_score_correction_bias";

const TEXT_FIXTURE_SHARD_A: &str = "fixture-a.safetensors";

const TEXT_FIXTURE_SHARD_B: &str = "fixture-b.safetensors";

static TEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct TextFixtureDirectory(PathBuf);

impl Drop for TextFixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct TextFixture {
    directory: TextFixtureDirectory,
    manifest: PackedArtifactManifest,
    limits: PackedSafetensorsLimits,
}

fn fixture_tensor(name: &str) -> (&str, Vec<usize>, Vec<u8>) {
    match name {
        TEXT_FIXTURE_WEIGHT | TEXT_FIXTURE_MTP => ("F8_E4M3", vec![1, 1], vec![1]),
        TEXT_FIXTURE_SCALE | TEXT_FIXTURE_MTP_SCALE => {
            ("F32", vec![1, 1], 1.0f32.to_le_bytes().to_vec())
        }
        TEXT_FIXTURE_VISION => ("U8", vec![1], vec![7]),
        TEXT_FIXTURE_BF16 => ("BF16", vec![1], vec![0, 0]),
        TEXT_FIXTURE_F32 => ("F32", vec![1], 0.0f32.to_le_bytes().to_vec()),
        other => panic!("unknown text fixture tensor {other}"),
    }
}

fn write_fixture_shard(root: &Path, filename: &str, names: &[&str]) -> PackedShardManifest {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for name in names {
        let (dtype, shape, bytes) = fixture_tensor(name);
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut shard = Vec::with_capacity(8 + header.len() + data.len());
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(&header);
    shard.extend_from_slice(&data);
    fs::write(root.join(filename), &shard).unwrap();
    PackedShardManifest {
        filename: filename.to_string(),
        file_length: shard.len(),
        file_sha256: sha256_digest(&shard),
        header_length: header.len(),
        header_sha256: sha256_digest(&header),
    }
}

fn text_fixture(redistribute: bool, revision: &str) -> TextFixture {
    let id = TEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("poot-glm53-card368-{}-{id}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let (a, b) = if redistribute {
        (
            vec![TEXT_FIXTURE_BF16, TEXT_FIXTURE_MTP, TEXT_FIXTURE_MTP_SCALE],
            vec![
                TEXT_FIXTURE_F32,
                TEXT_FIXTURE_SCALE,
                TEXT_FIXTURE_VISION,
                TEXT_FIXTURE_WEIGHT,
            ],
        )
    } else {
        (
            vec![TEXT_FIXTURE_BF16, TEXT_FIXTURE_SCALE, TEXT_FIXTURE_WEIGHT],
            vec![
                TEXT_FIXTURE_F32,
                TEXT_FIXTURE_MTP,
                TEXT_FIXTURE_MTP_SCALE,
                TEXT_FIXTURE_VISION,
            ],
        )
    };
    let shards = vec![
        write_fixture_shard(&root, TEXT_FIXTURE_SHARD_A, &a),
        write_fixture_shard(&root, TEXT_FIXTURE_SHARD_B, &b),
    ];
    let weight_map = a
        .iter()
        .map(|name| ((*name).to_string(), TEXT_FIXTURE_SHARD_A))
        .chain(
            b.iter()
                .map(|name| ((*name).to_string(), TEXT_FIXTURE_SHARD_B)),
        )
        .collect::<BTreeMap<_, _>>();
    let config = br#"{}"#;
    let index = serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap();
    fs::write(root.join("config.json"), config).unwrap();
    fs::write(root.join("model.safetensors.index.json"), &index).unwrap();
    let selected_source_bytes = 2 + 1 + 4 + 4;
    TextFixture {
        directory: TextFixtureDirectory(root),
        manifest: PackedArtifactManifest {
            repository: "local/glm53-card368".to_string(),
            revision: revision.to_string(),
            config_length: config.len(),
            config_sha256: sha256_digest(config),
            index_sha256: sha256_digest(&index),
            shards,
        },
        limits: PackedSafetensorsLimits {
            config_bytes: config.len(),
            index_bytes: index.len(),
            header_bytes_per_shard: 4_096,
            shard_count: 2,
            tensor_entries: 7,
            selected_source_bytes,
            packed_source_bytes: 5,
        },
    }
}

fn fixture_artifact_contract(manifest: &PackedArtifactManifest) -> Glm53FlashTextArtifactContract {
    Glm53FlashTextArtifactContract {
        repository: manifest.repository.clone(),
        revision: manifest.revision.clone(),
        config_sha256: sha256_digest_hex(manifest.config_sha256),
        index_sha256: sha256_digest_hex(manifest.index_sha256),
        shards: manifest
            .shards
            .iter()
            .map(|shard| Glm53FlashExpectedShard {
                name: shard.filename.clone(),
                size: shard.file_length,
                sha256: sha256_digest_hex(shard.file_sha256),
                header_length: shard.header_length,
                header_sha256: sha256_digest_hex(shard.header_sha256),
            })
            .collect(),
    }
}

fn fixture_classifier(manifest: &PackedArtifactManifest) -> Glm53FlashTextClassifier {
    let shard = |name: &str| -> &'static str {
        if name == TEXT_FIXTURE_BF16 || name == TEXT_FIXTURE_SCALE || name == TEXT_FIXTURE_WEIGHT {
            TEXT_FIXTURE_SHARD_A
        } else {
            TEXT_FIXTURE_SHARD_B
        }
    };
    let selected = |name: &str| {
        let (dtype, shape, logical_shape, quant_role) = match name {
            TEXT_FIXTURE_WEIGHT => ("F8_E4M3", vec![1, 1], vec![1, 1], "e4m3_weight"),
            TEXT_FIXTURE_SCALE => ("F32", vec![1, 1], vec![1, 1], "f32_block_scale"),
            TEXT_FIXTURE_BF16 => ("BF16", vec![1], vec![1], "unquantized"),
            TEXT_FIXTURE_F32 => ("F32", vec![1], vec![1], "unquantized"),
            _ => {
                return Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
                    "scoped fixture row reached selected descriptor validation: {name}"
                )));
            }
        };
        Ok(Glm53FlashExpectedTextKind::Selected {
            dtype: dtype.to_string(),
            shape,
            logical_shape,
            quant_role: quant_role.to_string(),
        })
    };
    let mut rows = [
        TEXT_FIXTURE_WEIGHT,
        TEXT_FIXTURE_SCALE,
        TEXT_FIXTURE_MTP,
        TEXT_FIXTURE_MTP_SCALE,
        TEXT_FIXTURE_VISION,
        TEXT_FIXTURE_BF16,
        TEXT_FIXTURE_F32,
    ]
    .into_iter()
    .map(|name| {
        Ok(Glm53FlashExpectedTextRow {
            name: name.to_string(),
            shard: shard(name).to_string(),
            kind: glm53_flash_expected_text_kind(name, || selected(name))?,
        })
    })
    .collect::<Result<Vec<_>, Glm53FlashMetadataError>>()
    .unwrap();
    rows.sort_by(|left, right| left.name.cmp(&right.name));
    let report = Glm53FlashTextInventoryReport {
        total_names: 7,
        selected_names: 4,
        deferred_mtp_names: 2,
        excluded_vision_names: 1,
        packed_pairs: 1,
        dsa_packed_pairs: 1,
        dense_bf16_names: 1,
        dense_f32_names: 1,
        ..Glm53FlashTextInventoryReport::default()
    };
    glm53_flash_text_report(&rows, report).unwrap();
    Glm53FlashTextClassifier {
        contract: Glm53FlashTextContract {
            artifact: fixture_artifact_contract(manifest),
            rows,
            report,
        },
    }
}

fn authenticate_text_fixture(fixture: &TextFixture) -> AuthenticatedSafetensorsHandleSet {
    AuthenticatedSafetensorsHandleSet::authenticate(
        &fixture.directory.0,
        fixture.manifest.clone(),
        fixture.limits,
    )
    .unwrap()
}

fn assert_empty_text_caches(packed: &PackedOwnerCache, exact: &ExactSourceOwnerCache) {
    assert!(packed.is_empty());
    assert!(exact.is_empty());
}

#[test]
fn glm5next_text_public_classifier_loads_exact_partition_and_defers_layer_45() {
    let fixture = text_fixture(false, "fixture-revision");
    let classifier = fixture_classifier(&fixture.manifest);
    let mut authenticated = authenticate_text_fixture(&fixture);
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let loaded = classifier
        .load(&mut authenticated, &mut packed_cache, &mut exact_cache)
        .unwrap();
    assert_eq!(loaded.report(), classifier.report());
    assert_eq!(loaded.mixed().packed_metadata.len(), 1);
    assert_eq!(loaded.mixed().exact_metadata.len(), 2);
    let mtp = loaded
        .mixed()
        .inventory
        .iter()
        .find(|row| row.descriptor.name() == TEXT_FIXTURE_MTP)
        .unwrap();
    assert_eq!(mtp.descriptor.dtype(), "F8_E4M3");
    assert_eq!(mtp.disposition, TensorDisposition::Deferred);
    let vision = loaded
        .mixed()
        .inventory
        .iter()
        .find(|row| row.descriptor.name() == TEXT_FIXTURE_VISION)
        .unwrap();
    assert_eq!(vision.descriptor.dtype(), "U8");
    assert_eq!(vision.disposition, TensorDisposition::Excluded);
}

#[test]
fn glm5next_text_public_classifier_rejects_wrong_snapshot_before_publication() {
    let fixture = text_fixture(false, "foreign-revision");
    let mut expected_manifest = fixture.manifest.clone();
    expected_manifest.revision = "fixture-revision".to_string();
    let classifier = fixture_classifier(&expected_manifest);
    let mut authenticated = authenticate_text_fixture(&fixture);
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = classifier
        .load(&mut authenticated, &mut packed_cache, &mut exact_cache)
        .unwrap_err()
        .to_string();
    assert!(error.contains("revision"), "{error}");
    assert_empty_text_caches(&packed_cache, &exact_cache);
}

#[test]
fn glm5next_text_transaction_rejects_foreign_session_keys_before_publication() {
    let fixture = text_fixture(false, "fixture-revision");
    let classifier = fixture_classifier(&fixture.manifest);
    let mut authenticated = authenticate_text_fixture(&fixture);
    let foreign = authenticate_text_fixture(&fixture);
    let foreign_decisions =
        classify_glm53_flash_text_inventory(&classifier.contract.rows, foreign.inventory())
            .unwrap();
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = classifier
        .load_with_decision_mutation(
            &mut authenticated,
            &mut packed_cache,
            &mut exact_cache,
            |decisions| *decisions = foreign_decisions,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("another authenticated session"), "{error}");
    assert_empty_text_caches(&packed_cache, &exact_cache);
}

#[test]
fn glm5next_text_public_classifier_rejects_ordinal_and_shard_redistribution() {
    let fixture = text_fixture(false, "fixture-revision");
    let mut classifier = fixture_classifier(&fixture.manifest);
    classifier.contract.rows.swap(0, 1);
    let mut authenticated = authenticate_text_fixture(&fixture);
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = classifier
        .load(&mut authenticated, &mut packed_cache, &mut exact_cache)
        .unwrap_err()
        .to_string();
    assert!(error.contains("has name"), "{error}");
    assert_empty_text_caches(&packed_cache, &exact_cache);

    let fixture = text_fixture(true, "fixture-revision");
    let mut classifier = fixture_classifier(&fixture.manifest);
    for row in &mut classifier.contract.rows {
        row.shard = if row.name == TEXT_FIXTURE_BF16
            || row.name == TEXT_FIXTURE_SCALE
            || row.name == TEXT_FIXTURE_WEIGHT
        {
            TEXT_FIXTURE_SHARD_A.to_string()
        } else {
            TEXT_FIXTURE_SHARD_B.to_string()
        };
    }
    let mut authenticated = authenticate_text_fixture(&fixture);
    let mut packed_cache = PackedOwnerCache::new();
    let mut exact_cache = ExactSourceOwnerCache::new();
    let error = classifier
        .load(&mut authenticated, &mut packed_cache, &mut exact_cache)
        .unwrap_err()
        .to_string();
    assert!(error.contains("has shard"), "{error}");
    assert_empty_text_caches(&packed_cache, &exact_cache);
}

#[test]
fn glm5next_text_transaction_rejects_missing_duplicate_and_pair_mutations() {
    for (case, mutate, expected) in [
        (
            "missing",
            (|decisions: &mut Vec<InventoryDecision>| {
                let _ = decisions.pop();
            }) as fn(&mut Vec<InventoryDecision>),
            "has no disposition",
        ),
        (
            "duplicate",
            |decisions: &mut Vec<InventoryDecision>| {
                decisions.push(decisions[0].clone());
            },
            "more than once",
        ),
        (
            "pair",
            |decisions: &mut Vec<InventoryDecision>| {
                let scale = decisions
                    .iter_mut()
                    .find(|decision| {
                        matches!(&decision.disposition, TensorDisposition::PackedScale { .. })
                    })
                    .unwrap();
                scale.disposition = TensorDisposition::Deferred;
            },
            "missing",
        ),
    ] {
        let fixture = text_fixture(false, "fixture-revision");
        let classifier = fixture_classifier(&fixture.manifest);
        let mut authenticated = authenticate_text_fixture(&fixture);
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let error = classifier
            .load_with_decision_mutation(
                &mut authenticated,
                &mut packed_cache,
                &mut exact_cache,
                mutate,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{case}: {error}");
        assert_empty_text_caches(&packed_cache, &exact_cache);
    }
}
