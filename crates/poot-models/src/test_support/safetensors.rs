//! Tiny authenticated safetensors checkpoints for model tests that bind Card 359 source owners.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use poot_load::packed_safetensors::{
    AuthenticatedSafetensorsHandleSet, ExactSourceOwnerCache, InventoryDecision, MixedLoadResult,
    PackedArtifactManifest, PackedOwnerCache, PackedSafetensorsLimits, PackedShardManifest,
    TensorDisposition, sha256_digest,
};
use serde_json::json;

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// A temporary directory removed on drop.
pub(crate) struct TempDir {
    pub(crate) path: PathBuf,
}

impl TempDir {
    pub(crate) fn new() -> Self {
        let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "poot-models-safetensors-{}-{ordinal}",
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

/// One safetensors tensor with its raw little-endian bytes.
pub(crate) struct SourceRow {
    pub(crate) name: String,
    pub(crate) dtype: &'static str,
    pub(crate) shape: Vec<usize>,
    pub(crate) bytes: Vec<u8>,
}

/// Write one safetensors shard holding `rows` in order.
pub(crate) fn write_shard(root: &Path, filename: &str, rows: &[SourceRow]) -> PackedShardManifest {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for row in rows {
        let start = data.len();
        data.extend_from_slice(&row.bytes);
        header.insert(
            row.name.clone(),
            json!({
                "dtype": row.dtype,
                "shape": row.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    fs::write(root.join(filename), &file).unwrap();
    PackedShardManifest {
        filename: filename.to_string(),
        file_length: file.len(),
        file_sha256: sha256_digest(&file),
        header_length: header.len(),
        header_sha256: sha256_digest(&header),
    }
}

/// A loaded Card 359 transaction and the temporary checkpoint directory it authenticated.
pub(crate) struct LoadedCheckpoint {
    _root: TempDir,
    pub(crate) mixed: MixedLoadResult,
}

/// Write `rows` as a one-shard checkpoint of `repository@revision`, authenticate it, and load every row with its
/// disposition.
pub(crate) fn load_checkpoint(
    repository: &str,
    revision: &str,
    config: &[u8],
    rows: Vec<(SourceRow, TensorDisposition)>,
) -> LoadedCheckpoint {
    let root = TempDir::new();
    fs::write(root.path.join("config.json"), config).unwrap();
    let filename = "model-00001-of-00001.safetensors";
    let (rows, dispositions): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    let shard = write_shard(&root.path, filename, &rows);
    let weight_map = rows
        .iter()
        .map(|row| (row.name.clone(), json!(filename)))
        .collect::<serde_json::Map<_, _>>();
    let index = serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap();
    fs::write(root.path.join("model.safetensors.index.json"), &index).unwrap();
    let manifest = PackedArtifactManifest {
        repository: repository.to_string(),
        revision: revision.to_string(),
        config_length: config.len(),
        config_sha256: sha256_digest(config),
        index_sha256: sha256_digest(&index),
        shards: vec![shard],
    };
    let mut authenticated = AuthenticatedSafetensorsHandleSet::authenticate(
        &root.path,
        manifest,
        PackedSafetensorsLimits {
            config_bytes: 4_096,
            index_bytes: 65_536,
            header_bytes_per_shard: 65_536,
            shard_count: 1,
            tensor_entries: rows.len(),
            selected_source_bytes: 65_536,
            packed_source_bytes: 65_536,
        },
    )
    .unwrap();
    let dispositions = rows
        .iter()
        .map(|row| row.name.as_str())
        .zip(dispositions)
        .collect::<HashMap<_, _>>();
    let mixed = authenticated
        .load_mixed(
            &mut PackedOwnerCache::new(),
            &mut ExactSourceOwnerCache::new(),
            |inventory| {
                inventory
                    .rows()
                    .map(|row| {
                        dispositions
                            .get(row.name())
                            .cloned()
                            .map(|disposition| InventoryDecision::new(row.key(), disposition))
                            .ok_or("unclassified checkpoint row")
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )
        .unwrap();
    LoadedCheckpoint { _root: root, mixed }
}

/// Little-endian BF16 bytes of `values`. Each value must be exactly representable in BF16, so no rounding hides in
/// the fixture.
pub(crate) fn bf16_bytes(name: &str, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            assert_eq!(
                bits & 0xffff,
                0,
                "{name} value {value} is not exactly representable in BF16"
            );
            u16::try_from(bits >> 16)
                .expect("the top half of an f32 fits a u16")
                .to_le_bytes()
        })
        .collect()
}

/// The BF16 value a checkpoint stores for `value`: its sign, exponent, and top seven mantissa bits.
pub(crate) fn truncate_to_bf16(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xffff_0000)
}
