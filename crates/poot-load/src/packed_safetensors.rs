//! Bounded, authenticated loading of selected packed safetensors pairs.
//!
//! This module deliberately does not use the generic dense checkpoint path. It retains the
//! files admitted by structural capability - a manifest, when the caller supplies one, is
//! optional integrity data verified once by [`validation::admit_file`]/[`validation::admit_shard`]
//! (ADR-0103 decision 4) - validates the complete shard topology before selection, and constructs
//! [`PackedPayload`] owners directly from exact source bytes.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use std::fmt;

use std::fs::File;

use std::hash::{Hash, Hasher};

use std::io::{Read, Seek, SeekFrom};

use std::path::{Component, Path};

use std::sync::Arc;

use poot_quant::format::WeightFormat;
use poot_quant::{
    MapError, OperandRole, PackedPayload, PackedWeight, PackedWeightError, SliceError, SourceRole,
    StoredBytes,
};

use ring::digest::{Context, SHA256};

use serde::de::{self, MapAccess, SeqAccess, Visitor};

use serde::{Deserialize, Deserializer};

mod inventory;
mod publication;
mod store;
mod validation;

pub use inventory::*;
pub(crate) use publication::*;
pub use store::*;
pub(crate) use validation::*;
pub use validation::{FilePin, check_pin_reader, parse_json_unique_keys};

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::rc::Rc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    const SHARD: &str = "model-00001-of-00001.safetensors";

    #[derive(Clone, Default)]
    struct ReadCount(Rc<Cell<usize>>);

    impl ReadCount {
        fn get(&self) -> usize {
            self.0.get()
        }
    }

    struct CountingReader {
        bytes: Cursor<Vec<u8>>,
        read_count: ReadCount,
    }

    impl CountingReader {
        fn new(bytes: Vec<u8>, read_count: ReadCount) -> Self {
            Self {
                bytes: Cursor::new(bytes),
                read_count,
            }
        }
    }

    impl Read for CountingReader {
        fn read(&mut self, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
            let read = self.bytes.read(buffer)?;
            self.read_count.0.set(self.read_count.get() + read);
            Ok(read)
        }
    }

    impl Seek for CountingReader {
        fn seek(&mut self, position: SeekFrom) -> Result<u64, std::io::Error> {
            self.bytes.seek(position)
        }
    }

    impl RetainedReader for CountingReader {
        fn file_length(&self) -> Result<u64, std::io::Error> {
            Ok(self.bytes.get_ref().len() as u64)
        }
    }

    struct CountingFixture {
        manifest: PackedArtifactManifest,
        limits: PackedSafetensorsLimits,
        config: Vec<u8>,
        index: Vec<u8>,
        shard: Vec<u8>,
    }

    struct FixtureReads {
        config: ReadCount,
        index: ReadCount,
    }

    impl CountingFixture {
        fn new() -> Self {
            let config = br#"{}"#.to_vec();
            let index = format!(r#"{{"weight_map":{{"weight":"{SHARD}"}}}}"#).into_bytes();
            let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
            let mut shard = Vec::with_capacity(8 + header.len() + 1);
            shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
            shard.extend_from_slice(header);
            shard.push(7);
            Self {
                manifest: PackedArtifactManifest {
                    repository: "local/counting".to_string(),
                    revision: "counting-revision".to_string(),
                    config_length: config.len(),
                    config_sha256: sha256_digest(&config),
                    index_sha256: sha256_digest(&index),
                    shards: vec![PackedShardManifest {
                        filename: SHARD.to_string(),
                        file_length: shard.len(),
                        file_sha256: sha256_digest(&shard),
                        header_length: header.len(),
                        header_sha256: sha256_digest(header),
                    }],
                },
                limits: PackedSafetensorsLimits {
                    config_bytes: config.len(),
                    index_bytes: index.len(),
                    header_bytes_per_shard: header.len(),
                    shard_count: 1,
                    tensor_entries: 1,
                    selected_source_bytes: 1,
                    packed_source_bytes: 1,
                },
                config,
                index,
                shard,
            }
        }

        fn authenticate(
            &self,
            manifest: PackedArtifactManifest,
            limits: PackedSafetensorsLimits,
        ) -> (
            Result<AuthenticatedSafetensorsHandleSet, PackedSafetensorsError>,
            FixtureReads,
        ) {
            let shard_manifest = manifest.shards[0].clone();
            let reads = FixtureReads {
                config: ReadCount::default(),
                index: ReadCount::default(),
            };
            let opened_shards = BTreeMap::from([(
                SHARD.to_string(),
                (Some(shard_manifest), StoredBytes::from(self.shard.clone())),
            )]);
            let result = AuthenticatedSafetensorsHandleSet::authenticate_retained(
                manifest,
                limits,
                Box::new(CountingReader::new(
                    self.config.clone(),
                    reads.config.clone(),
                )),
                Box::new(CountingReader::new(self.index.clone(), reads.index.clone())),
                opened_shards,
            );
            (result, reads)
        }
    }

    fn assert_reads(reads: &FixtureReads, config: usize, index: usize) {
        assert_eq!(reads.config.get(), config, "config bytes read");
        assert_eq!(reads.index.get(), index, "index bytes read");
    }

    #[test]
    fn authentication_retains_the_verified_config_bytes() {
        let fixture = CountingFixture::new();
        let (authenticated, reads) = fixture.authenticate(fixture.manifest.clone(), fixture.limits);
        let authenticated = authenticated.expect("fixture authenticates");

        assert_eq!(authenticated.config_bytes(), fixture.config);
        // The shard is a mapping, not a reader: only the small files are read, once each.
        assert_reads(&reads, fixture.config.len(), fixture.index.len());
    }

    #[test]
    fn authentication_retains_the_supplied_config_owner_without_rereading_it() {
        let fixture = CountingFixture::new();
        let shard_manifest = fixture.manifest.shards[0].clone();
        let config = fixture.config.clone();
        let config_owner = config.as_ptr();
        let config_reads = ReadCount::default();
        let index_reads = ReadCount::default();
        let authenticated = AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
            ArtifactLabel {
                repository: fixture.manifest.repository.clone(),
                revision: fixture.manifest.revision.clone(),
            },
            Some(fixture.manifest.clone()),
            fixture.limits,
            Box::new(CountingReader::new(
                fixture.config.clone(),
                config_reads.clone(),
            )),
            config,
            Box::new(CountingReader::new(
                fixture.index.clone(),
                index_reads.clone(),
            )),
            BTreeMap::from([(
                SHARD.to_string(),
                (
                    Some(shard_manifest),
                    StoredBytes::from(fixture.shard.clone()),
                ),
            )]),
        )
        .expect("fixture authenticates from its retained config owner");

        assert_eq!(authenticated.config_bytes().as_ptr(), config_owner);
        assert_eq!(config_reads.get(), 0, "config handle must not be reread");
        assert_eq!(index_reads.get(), fixture.index.len());
    }

    #[test]
    fn retained_config_bridge_preserves_the_dispatch_allocation() {
        static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

        let fixture = CountingFixture::new();
        let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "poot-retained-config-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(CONFIG_FILE), &fixture.config).unwrap();
        std::fs::write(root.join(INDEX_FILE), &fixture.index).unwrap();
        std::fs::write(root.join(SHARD), &fixture.shard).unwrap();

        let retained = retain_config_for_authentication(&root, fixture.limits.config_bytes)
            .expect("bounded config retention succeeds");
        let dispatch_owner = retained.bytes().as_ptr();
        assert_eq!(retained.pre_read_length(), fixture.config.len());
        assert!(retained.pre_read_regular());
        let authenticated = AuthenticatedSafetensorsHandleSet::authenticate_with_retained_config(
            &root,
            fixture.manifest.clone(),
            fixture.limits,
            retained,
        )
        .expect("retained config authenticates");
        assert_eq!(authenticated.config_bytes().as_ptr(), dispatch_owner);
        assert_eq!(authenticated.index_bytes.len(), fixture.index.len());

        std::fs::remove_dir_all(root).unwrap();
    }

    /// Card 544 SC-001 (first test): a shard with no manifest pin loads by capability alone -
    /// its declared shapes are valid, so a changed data byte does not refuse it.
    ///
    /// Mutation: change `manifest: None` to `manifest: Some(fixture.manifest.clone())`; the
    /// `.expect(...)` below then panics on the pinned digest mismatch, which is exactly
    /// `card544_sc001_manifest_refuses_a_shard_whose_digest_no_longer_matches`'s row.
    #[test]
    fn card544_sc001_capability_admits_a_tampered_shard_with_no_manifest() {
        let fixture = CountingFixture::new();
        let mut tampered_shard = fixture.shard.clone();
        let last = tampered_shard.len() - 1;
        tampered_shard[last] = tampered_shard[last].wrapping_add(1);
        assert_ne!(
            tampered_shard, fixture.shard,
            "mutation changes the shard's data byte"
        );
        assert_eq!(
            tampered_shard.len(),
            fixture.shard.len(),
            "mutation preserves length and the shard's declared shape"
        );

        let opened_shards =
            BTreeMap::from([(SHARD.to_string(), (None, StoredBytes::from(tampered_shard)))]);
        let authenticated = AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
            ArtifactLabel {
                repository: fixture.manifest.repository.clone(),
                revision: fixture.manifest.revision.clone(),
            },
            None,
            fixture.limits,
            Box::new(CountingReader::new(
                fixture.config.clone(),
                ReadCount::default(),
            )),
            fixture.config.clone(),
            Box::new(CountingReader::new(
                fixture.index.clone(),
                ReadCount::default(),
            )),
            opened_shards,
        )
        .expect("a tampered shard with no manifest pin loads by capability alone");
        authenticated
            .verify_before_publication()
            .expect("publication admits the same capability-only shard, not just authentication");
    }

    /// Card 544 SC-001 (second test): a shard whose manifest pin no longer matches its content is
    /// refused. The tampered byte is in the tensor data, not the header, so structural
    /// authentication still succeeds; the pin catches it at the one full-content verification
    /// pass, before publication (never before, and never twice - card 544 SC-002).
    ///
    /// Mutation: skip the digest comparison by passing `manifest: None`; the refusal below goes
    /// red, which is exactly `card544_sc001_capability_admits_a_tampered_shard_with_no_manifest`'s
    /// row.
    #[test]
    fn card544_sc001_manifest_refuses_a_shard_whose_digest_no_longer_matches() {
        let fixture = CountingFixture::new();
        let mut tampered_shard = fixture.shard.clone();
        let last = tampered_shard.len() - 1;
        tampered_shard[last] = tampered_shard[last].wrapping_add(1);
        let shard_manifest = fixture.manifest.shards[0].clone();

        let opened_shards = BTreeMap::from([(
            SHARD.to_string(),
            (Some(shard_manifest), StoredBytes::from(tampered_shard)),
        )]);
        let authenticated = AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
            ArtifactLabel {
                repository: fixture.manifest.repository.clone(),
                revision: fixture.manifest.revision.clone(),
            },
            Some(fixture.manifest.clone()),
            fixture.limits,
            Box::new(CountingReader::new(
                fixture.config.clone(),
                ReadCount::default(),
            )),
            fixture.config.clone(),
            Box::new(CountingReader::new(
                fixture.index.clone(),
                ReadCount::default(),
            )),
            opened_shards,
        )
        .expect("the header is untouched, so structural authentication still succeeds");
        let error = authenticated
            .verify_before_publication()
            .expect_err("a pinned digest that no longer matches the shard must refuse the load");
        assert!(
            matches!(error, PackedSafetensorsError::DigestMismatch { .. }),
            "{error:?}"
        );
    }

    /// Card 544 SC-002 (ADR-0103 amendment): a shard is hashed at most once across a whole load,
    /// over its mapping, and only when the manifest pins it. Authentication hashes none of the
    /// body (only the small files); publication hashes a pinned shard exactly once and an unpinned
    /// one never. The old streamed-read count (header prefix plus one pass) is retired with the
    /// reader: the mapping has no read calls to count.
    ///
    /// Mutation: hash the pinned shard twice in `verify_before_publication`; the first assertion's
    /// byte count doubles and goes red.
    #[test]
    fn card544_sc002_publication_hashes_a_pinned_shard_once_and_an_unpinned_one_never() {
        let fixture = CountingFixture::new();
        let authenticate = |manifest: Option<PackedArtifactManifest>| {
            AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
                ArtifactLabel {
                    repository: fixture.manifest.repository.clone(),
                    revision: fixture.manifest.revision.clone(),
                },
                manifest.clone(),
                fixture.limits,
                Box::new(CountingReader::new(
                    fixture.config.clone(),
                    ReadCount::default(),
                )),
                fixture.config.clone(),
                Box::new(CountingReader::new(
                    fixture.index.clone(),
                    ReadCount::default(),
                )),
                BTreeMap::from([(
                    SHARD.to_string(),
                    (
                        manifest.map(|manifest| manifest.shards[0].clone()),
                        StoredBytes::from(fixture.shard.clone()),
                    ),
                )]),
            )
            .expect("fixture authenticates")
        };

        assert_eq!(
            authenticate(Some(fixture.manifest.clone()))
                .verify_before_publication()
                .expect("the unmutated shard matches its pin"),
            fixture.shard.len(),
            "a pinned shard is hashed once, over its whole mapping"
        );
        assert_eq!(
            authenticate(None)
                .verify_before_publication()
                .expect("an unpinned shard is admitted by capability"),
            0,
            "an unpinned shard is not hashed at all"
        );
    }

    /// A scratch directory holding real shard files, removed on drop. Fixtures map their shards
    /// through the production open path ([`map_shard`]), so a test can change the file under the
    /// mapping exactly as an outside writer would.
    struct TempRoot(std::path::PathBuf);

    impl TempRoot {
        fn new(label: &str) -> Self {
            static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);
            let serial = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
            let root =
                std::env::temp_dir().join(format!("poot-{label}-{}-{serial}", std::process::id()));
            std::fs::create_dir(&root).unwrap();
            Self(root)
        }

        fn write(&self, name: &str, bytes: &[u8]) {
            std::fs::write(self.0.join(name), bytes).unwrap();
        }

        /// Truncate `name` to zero length through a second handle. The mapping already open over
        /// it must not be read after this: its pages are gone.
        fn truncate(&self, name: &str) {
            std::fs::OpenOptions::new()
                .write(true)
                .open(self.0.join(name))
                .unwrap()
                .set_len(0)
                .unwrap();
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn rollback_fixture() -> (
        AuthenticatedSafetensorsHandleSet,
        Vec<PackedSelectionRow>,
        TempRoot,
    ) {
        let config = br#"{}"#;
        let index = format!(
            r#"{{"weight_map":{{"first.scale":"{SHARD}","first.weight":"{SHARD}","second.scale":"{SHARD}","second.weight":"{SHARD}"}}}}"#
        )
        .into_bytes();
        let header = serde_json::to_vec(&serde_json::json!({
            "first.weight": {"dtype": "F8_E4M3", "shape": [1, 1], "data_offsets": [0, 1]},
            "first.scale": {"dtype": "F32", "shape": [1, 1], "data_offsets": [1, 5]},
            "second.weight": {"dtype": "F8_E4M3", "shape": [1, 1], "data_offsets": [5, 6]},
            "second.scale": {"dtype": "F32", "shape": [1, 1], "data_offsets": [6, 10]},
        }))
        .expect("synthetic header serializes");
        let second_weight = 0x7f;
        let data = [
            0x01,
            0x00,
            0x00,
            0x80,
            0x3f,
            second_weight,
            0x00,
            0x00,
            0x80,
            0x3f,
        ];
        let mut shard = Vec::with_capacity(8 + header.len() + data.len());
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(&header);
        shard.extend_from_slice(&data);
        let shard_manifest = PackedShardManifest {
            filename: SHARD.to_string(),
            file_length: shard.len(),
            file_sha256: sha256_digest(&shard),
            header_length: header.len(),
            header_sha256: sha256_digest(&header),
        };
        let root = TempRoot::new("rollback");
        root.write(SHARD, &shard);
        let manifest = PackedArtifactManifest {
            repository: "local/rollback".to_string(),
            revision: "rollback-revision".to_string(),
            config_length: config.len(),
            config_sha256: sha256_digest(config),
            index_sha256: sha256_digest(&index),
            shards: vec![shard_manifest.clone()],
        };
        let limits = PackedSafetensorsLimits {
            config_bytes: config.len(),
            index_bytes: index.len(),
            header_bytes_per_shard: header.len(),
            shard_count: 1,
            tensor_entries: 4,
            selected_source_bytes: 10,
            packed_source_bytes: 10,
        };
        let opened_shards = BTreeMap::from([(
            SHARD.to_string(),
            (
                Some(shard_manifest),
                map_shard(&root.0, SHARD).expect("the fixture shard maps"),
            ),
        )]);
        let authenticated = AuthenticatedSafetensorsHandleSet::authenticate_retained(
            manifest,
            limits,
            Box::new(CountingReader::new(config.to_vec(), ReadCount::default())),
            Box::new(CountingReader::new(index, ReadCount::default())),
            opened_shards,
        )
        .unwrap_or_else(|error| panic!("rollback fixture authentication failed: {error}"));
        let descriptor = PackedWeight::try_new(
            WeightFormat::E4m3Block128 {
                scale: poot_quant::format::ScaleEncoding::F32,
            },
            [1, 1],
        )
        .expect("rollback descriptor is valid");
        let selections = vec![
            PackedSelectionRow {
                linear_id: "first".to_string(),
                descriptor,
                weight_name: "first.weight".to_string(),
                scale_name: "first.scale".to_string(),
                shard: SHARD.to_string(),
                weight_span: SourceSpan::new(0, 1),
                scale_span: SourceSpan::new(1, 5),
                weight_dtype: "F8_E4M3".to_string(),
                scale_dtype: "F32".to_string(),
                weight_shape: [1, 1],
                scale_shape: [1, 1],
            },
            PackedSelectionRow {
                linear_id: "second".to_string(),
                descriptor,
                weight_name: "second.weight".to_string(),
                scale_name: "second.scale".to_string(),
                shard: SHARD.to_string(),
                weight_span: SourceSpan::new(5, 6),
                scale_span: SourceSpan::new(6, 10),
                weight_dtype: "F8_E4M3".to_string(),
                scale_dtype: "F32".to_string(),
                weight_shape: [1, 1],
                scale_shape: [1, 1],
            },
        ];
        (authenticated, selections, root)
    }

    /// The second pair's payload fails its content check after the first pair is staged: the first
    /// staged owner is dropped and nothing is published. The mid-read I/O failure variant of this
    /// test is retired with the copy path: a mapped source has no read to fail.
    #[test]
    fn second_pair_failure_drops_actual_first_staged_owner() {
        let (mut authenticated, selections, _root) = rollback_fixture();
        let mut cache = PackedOwnerCache::new();
        let mut staged_owners = Vec::new();
        let result = authenticated.load_with_staging_observer(&selections, &mut cache, |owner| {
            staged_owners.push(Arc::downgrade(owner))
        });
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::PayloadBuild {
                completed_payloads: 1,
                ..
            })
        ));
        assert!(cache.is_empty());
        assert_eq!(staged_owners.len(), 1);
        assert!(staged_owners[0].upgrade().is_none());
    }

    fn exact_source_key_fixture() -> ExactSourceCacheKey {
        ExactSourceCacheKey {
            artifact: AuthenticatedArtifactIdentity(Arc::new(ArtifactIdentityFields {
                repository: "local/key-fixture".to_string(),
                revision: "key-revision".to_string(),
                config_length: 2,
                config_sha256: Sha256Digest::new([1; 32]),
                index_length: 3,
                index_sha256: Sha256Digest::new([2; 32]),
                shards: vec![PackedShardManifest {
                    filename: SHARD.to_string(),
                    file_length: 101,
                    file_sha256: Sha256Digest::new([3; 32]),
                    header_length: 59,
                    header_sha256: Sha256Digest::new([4; 32]),
                }],
            })),
            descriptor: AuthenticatedTensorDescriptor {
                name: "standalone".to_string(),
                shard: SHARD.to_string(),
                span: SourceSpan::new(5, 8),
                dtype: "F8_E4M3".to_string(),
                shape: vec![3],
            },
            kind: ExactSourceKind::E4m3,
        }
    }

    fn rollback_mixed_decisions(inventory: AuthenticatedInventory<'_>) -> Vec<InventoryDecision> {
        inventory
            .rows()
            .map(|row| {
                let disposition = match row.name() {
                    "first.weight" => TensorDisposition::PackedWeight {
                        linear_id: "first".to_string(),
                        format: WeightFormat::E4m3Block128 {
                            scale: poot_quant::format::ScaleEncoding::F32,
                        },
                        logical_shape: [1, 1],
                    },
                    "first.scale" => TensorDisposition::PackedScale {
                        linear_id: "first".to_string(),
                    },
                    "second.weight" => TensorDisposition::StandaloneE4m3,
                    "second.scale" => TensorDisposition::Deferred,
                    other => panic!("unexpected rollback row {other}"),
                };
                InventoryDecision::new(row.key(), disposition)
            })
            .collect()
    }

    #[test]
    fn exact_source_cache_key_covers_artifact_and_selection() {
        let expected = exact_source_key_fixture();

        let mutations: [fn(&mut ExactSourceCacheKey); 18] = [
            |key| Arc::make_mut(&mut key.artifact.0).repository.push('x'),
            |key| Arc::make_mut(&mut key.artifact.0).revision.push('x'),
            |key| Arc::make_mut(&mut key.artifact.0).config_length += 1,
            |key| {
                Arc::make_mut(&mut key.artifact.0).config_sha256 = Sha256Digest::new([5; 32]);
            },
            |key| Arc::make_mut(&mut key.artifact.0).index_length += 1,
            |key| {
                Arc::make_mut(&mut key.artifact.0).index_sha256 = Sha256Digest::new([6; 32]);
            },
            |key| {
                Arc::make_mut(&mut key.artifact.0).shards[0]
                    .filename
                    .push('x');
            },
            |key| Arc::make_mut(&mut key.artifact.0).shards[0].file_length += 1,
            |key| {
                Arc::make_mut(&mut key.artifact.0).shards[0].file_sha256 =
                    Sha256Digest::new([7; 32]);
            },
            |key| Arc::make_mut(&mut key.artifact.0).shards[0].header_length += 1,
            |key| {
                Arc::make_mut(&mut key.artifact.0).shards[0].header_sha256 =
                    Sha256Digest::new([8; 32]);
            },
            |key| key.descriptor.name.push('x'),
            |key| key.descriptor.shard.push('x'),
            |key| key.descriptor.span.start += 1,
            |key| key.descriptor.span.end += 1,
            |key| key.descriptor.dtype = "BF16".to_string(),
            |key| key.descriptor.shape[0] += 1,
            |key| key.kind = ExactSourceKind::Bf16,
        ];
        let set = HashSet::from([expected.clone()]);
        for mutation in mutations {
            let mut variant = exact_source_key_fixture();
            mutation(&mut variant);
            assert_ne!(variant, expected);
            assert!(!set.contains(&variant));
        }
    }

    /// A mixed load whose shard is truncated after every owner is staged is refused at the
    /// publication gate as `ArtifactChanged`, publishes nothing, drops every staged owner, and the
    /// process survives (no staged page is read after the truncation).
    #[test]
    fn mixed_snapshot_publication_is_atomic() {
        let (mut authenticated, _selections, root) = rollback_fixture();
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let mut packed_owners = Vec::new();
        let mut exact_owners = Vec::new();
        let result = authenticated.load_mixed_with_staging_observer(
            &mut packed_cache,
            &mut exact_cache,
            |inventory| Ok::<_, ()>(rollback_mixed_decisions(inventory)),
            |owner| match owner {
                StagedOwnerRef::Packed(owner) => packed_owners.push(Arc::downgrade(owner)),
                StagedOwnerRef::Exact { owner } => {
                    exact_owners.push(Arc::downgrade(owner));
                    root.truncate(SHARD);
                }
            },
        );
        assert!(matches!(
            result,
            Err(MixedLoadError::Assembly(
                PackedSafetensorsError::ArtifactChanged { .. }
            ))
        ));
        assert!(packed_cache.is_empty());
        assert!(exact_cache.is_empty());
        assert_eq!(packed_owners.len(), 1);
        assert!(packed_owners[0].upgrade().is_none());
        assert_eq!(exact_owners.len(), 1);
        assert!(exact_owners[0].upgrade().is_none());
    }

    /// An exact-source owner is a view of the shard's one mapping, not a copy of it. Replaces the
    /// retired read-allocation handoff test (`Arc<[u8]>` pointer and strong-count identity of a
    /// zero-filled read buffer), whose buffer no longer exists.
    #[test]
    fn exact_source_owner_is_a_view_of_the_shard_mapping() {
        let (mut authenticated, _selections, _root) = rollback_fixture();
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let result = authenticated
            .load_mixed_with_staging_observer(
                &mut packed_cache,
                &mut exact_cache,
                |inventory| Ok::<_, ()>(rollback_mixed_decisions(inventory)),
                |_| {},
            )
            .unwrap();

        assert_eq!(result.exact_metadata.len(), 1);
        let published = result.exact_metadata.values().next().unwrap();
        let shard = &authenticated.shards[SHARD];
        assert!(published.owner.bytes.same_region(&shard.bytes));
        assert_eq!(published.owner.bytes(), &[0x7f]);
    }

    #[derive(Clone, Copy, Debug)]
    enum PreProbeFailure {
        Classifier,
        ForeignKey,
        DuplicateKey,
        OmittedKey,
        DuplicateSelectedName,
        DuplicateEmptySpan,
    }

    #[test]
    fn mixed_validation_failures_precede_every_cache_probe() {
        for failure in [
            PreProbeFailure::Classifier,
            PreProbeFailure::ForeignKey,
            PreProbeFailure::DuplicateKey,
            PreProbeFailure::OmittedKey,
            PreProbeFailure::DuplicateSelectedName,
            PreProbeFailure::DuplicateEmptySpan,
        ] {
            let (mut authenticated, _selections, _root) = rollback_fixture();
            match failure {
                PreProbeFailure::DuplicateSelectedName => {
                    authenticated.inventory[3].name = authenticated.inventory[1].name.clone();
                }
                PreProbeFailure::DuplicateEmptySpan => {
                    for ordinal in [0, 2] {
                        authenticated.inventory[ordinal].span = SourceSpan::new(0, 0);
                        authenticated.inventory[ordinal].shape = vec![0];
                    }
                }
                _ => {}
            }
            let (foreign, _selections, _root) = rollback_fixture();
            let foreign_key = foreign.inventory().rows().next().unwrap().key();
            let mut packed_cache = PackedOwnerCache::new();
            let mut exact_cache = ExactSourceOwnerCache::new();
            let mut probes = 0usize;
            let mut constructions = 0usize;
            let mut staged = 0usize;
            let result = authenticated.load_mixed_with_observers(
                &mut packed_cache,
                &mut exact_cache,
                |inventory| {
                    if matches!(failure, PreProbeFailure::Classifier) {
                        return Err("classifier sentinel");
                    }
                    let mut decisions = match failure {
                        PreProbeFailure::DuplicateSelectedName => inventory
                            .rows()
                            .enumerate()
                            .map(|(ordinal, row)| {
                                let disposition = if matches!(ordinal, 1 | 3) {
                                    TensorDisposition::StandaloneE4m3
                                } else {
                                    TensorDisposition::Deferred
                                };
                                InventoryDecision::new(row.key(), disposition)
                            })
                            .collect(),
                        PreProbeFailure::DuplicateEmptySpan => inventory
                            .rows()
                            .enumerate()
                            .map(|(ordinal, row)| {
                                let disposition = if matches!(ordinal, 0 | 2) {
                                    TensorDisposition::DenseF32
                                } else {
                                    TensorDisposition::Deferred
                                };
                                InventoryDecision::new(row.key(), disposition)
                            })
                            .collect(),
                        _ => rollback_mixed_decisions(inventory),
                    };
                    match failure {
                        PreProbeFailure::Classifier => unreachable!(),
                        PreProbeFailure::ForeignKey => decisions.push(InventoryDecision::new(
                            foreign_key.clone(),
                            TensorDisposition::Deferred,
                        )),
                        PreProbeFailure::DuplicateKey => decisions.push(decisions[0].clone()),
                        PreProbeFailure::OmittedKey => {
                            decisions.pop();
                        }
                        PreProbeFailure::DuplicateSelectedName
                        | PreProbeFailure::DuplicateEmptySpan => {}
                    }
                    Ok(decisions)
                },
                |_| probes += 1,
                |_| constructions += 1,
                |_| staged += 1,
            );
            let expected_error = match failure {
                PreProbeFailure::Classifier => {
                    matches!(
                        result,
                        Err(MixedLoadError::Classifier("classifier sentinel"))
                    )
                }
                PreProbeFailure::ForeignKey => matches!(
                    result,
                    Err(MixedLoadError::Assembly(
                        PackedSafetensorsError::ForeignInventoryKey { .. }
                    ))
                ),
                PreProbeFailure::DuplicateKey => matches!(
                    result,
                    Err(MixedLoadError::Assembly(
                        PackedSafetensorsError::DuplicateInventoryKey { .. }
                    ))
                ),
                PreProbeFailure::OmittedKey => matches!(
                    result,
                    Err(MixedLoadError::Assembly(
                        PackedSafetensorsError::MissingInventoryDecision { .. }
                    ))
                ),
                PreProbeFailure::DuplicateSelectedName => matches!(
                    result,
                    Err(MixedLoadError::Assembly(
                        PackedSafetensorsError::ReusedSourceName { .. }
                    ))
                ),
                PreProbeFailure::DuplicateEmptySpan => matches!(
                    result,
                    Err(MixedLoadError::Assembly(
                        PackedSafetensorsError::ReusedSourceSpan {
                            span: SourceSpan { start: 0, end: 0 },
                            ..
                        }
                    ))
                ),
            };
            assert!(expected_error, "{failure:?} returned the wrong error");
            assert_eq!(probes, 0, "{failure:?} probed a cache");
            assert_eq!(constructions, 0, "{failure:?} constructed an owner");
            assert_eq!(staged, 0, "{failure:?} staged an owner");
            assert!(packed_cache.is_empty());
            assert!(exact_cache.is_empty());
        }
    }

    #[test]
    fn mixed_snapshot_accounting_is_checked() {
        for quantity in MixedByteQuantity::ADDITIVE {
            for category in MixedSourceCategory::ALL {
                let mut accounting = MixedAccountingTable::default();
                accounting.values[quantity as usize][category as usize] = usize::MAX;
                assert!(matches!(
                    accounting.checked_add(quantity, category, 1),
                    Err(PackedSafetensorsError::ArithmeticOverflow { field })
                        if field == quantity.field()
                ));
            }
        }

        for category in MixedSourceCategory::ALL {
            let mut accounting = MixedAccountingTable::default();
            accounting.values[MixedByteQuantity::NewOwner as usize][category as usize] = usize::MAX;
            accounting.values[MixedByteQuantity::ReusedOwner as usize][category as usize] = 1;
            assert!(matches!(
                accounting.derive_aggregate_owner(),
                Err(PackedSafetensorsError::ArithmeticOverflow { field })
                    if field == MixedByteQuantity::AggregateOwner.field()
            ));
        }

        for quantity in MixedByteQuantity::ALL {
            let mut accounting = MixedAccountingTable::default();
            accounting.values[quantity as usize][0] = usize::MAX;
            accounting.values[quantity as usize][1] = 1;
            if quantity == MixedByteQuantity::PeakInFlightRead {
                assert_eq!(
                    accounting.summarize(quantity).unwrap().aggregate(),
                    usize::MAX
                );
            } else {
                assert!(matches!(
                    accounting.summarize(quantity),
                    Err(PackedSafetensorsError::ArithmeticOverflow { field })
                        if field == quantity.field()
                ));
            }
        }

        let authentication = PackedLoadReport {
            config_bytes: usize::MAX,
            index_bytes: 1,
            ..PackedLoadReport::default()
        };
        assert!(matches!(
            mixed_metadata_bytes(authentication),
            Err(PackedSafetensorsError::ArithmeticOverflow {
                field: "mixed metadata bytes"
            })
        ));

        let mut packed_builds = usize::MAX;
        assert!(matches!(
            checked_increment_packed_builds(&mut packed_builds),
            Err(PackedSafetensorsError::ArithmeticOverflow {
                field: "mixed packed payload build count"
            })
        ));
        for kind in ExactSourceKind::ALL {
            let mut builds = ExactSourceBuildCounts::default();
            builds.values[kind as usize] = usize::MAX;
            assert!(matches!(
                checked_increment_exact_builds(&mut builds, kind),
                Err(PackedSafetensorsError::ArithmeticOverflow {
                    field: "exact-source build count"
                })
            ));
        }
        assert!(matches!(
            checked_sum(&[usize::MAX, 1, 0, 0], "exact-source build count total"),
            Err(PackedSafetensorsError::ArithmeticOverflow {
                field: "exact-source build count total"
            })
        ));
    }

    #[test]
    fn metadata_and_file_limits_precede_guarded_reads() {
        let fixture = CountingFixture::new();

        let mut limits = fixture.limits;
        limits.config_bytes -= 1;
        let (result, reads) = fixture.authenticate(fixture.manifest.clone(), limits);
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::LimitExceeded {
                kind: PackedLimitKind::ConfigBytes,
                ..
            })
        ));
        assert_reads(&reads, 0, 0);

        let mut limits = fixture.limits;
        limits.index_bytes -= 1;
        let (result, reads) = fixture.authenticate(fixture.manifest.clone(), limits);
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::LimitExceeded {
                kind: PackedLimitKind::IndexBytes,
                ..
            })
        ));
        assert_reads(&reads, fixture.config.len(), 0);

        let mut manifest = fixture.manifest.clone();
        manifest.shards[0].file_length += 1;
        let (result, reads) = fixture.authenticate(manifest, fixture.limits);
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::FileLengthMismatch { .. })
        ));
        assert_reads(&reads, fixture.config.len(), fixture.index.len());

        let mut limits = fixture.limits;
        limits.header_bytes_per_shard -= 1;
        let (result, reads) = fixture.authenticate(fixture.manifest.clone(), limits);
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::LimitExceeded {
                kind: PackedLimitKind::HeaderBytesPerShard,
                ..
            })
        ));
        assert_reads(&reads, fixture.config.len(), fixture.index.len());

        let mut manifest = fixture.manifest.clone();
        manifest.shards.push(manifest.shards[0].clone());
        let (result, reads) = fixture.authenticate(manifest, fixture.limits);
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::LimitExceeded {
                kind: PackedLimitKind::ShardCount,
                ..
            })
        ));
        assert_reads(&reads, 0, 0);
    }

    /// A shard truncated in place between the loader's span views and its publication gate is
    /// refused as `ArtifactChanged`, with no pin, and the process survives. Re-expresses the retired
    /// second-stream `RangeCheck` mismatch test (mutant H4: a range compared at the wrong buffer
    /// offset): a second streamed read of the same pages proves nothing, so the contract now rests
    /// on the mapping's file identity, checked once at publication. The untouched shard must load.
    ///
    /// Mutation: drop the `check_unchanged` call in `verify_before_publication`; the truncated
    /// shard is published and the refusal below goes red.
    #[test]
    fn a_shard_truncated_before_publication_is_refused_as_artifact_changed() {
        let config = br#"{}"#.to_vec();
        let index = format!(r#"{{"weight_map":{{"w":"{SHARD}","s":"{SHARD}"}}}}"#).into_bytes();
        let header = br#"{"w":{"dtype":"F8_E4M3","shape":[2,5],"data_offsets":[0,10]},"s":{"dtype":"F32","shape":[1,1],"data_offsets":[10,14]}}"#;
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(header);
        shard.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        shard.extend_from_slice(&1.0f32.to_le_bytes());
        let limits = PackedSafetensorsLimits {
            config_bytes: config.len(),
            index_bytes: index.len(),
            header_bytes_per_shard: header.len(),
            shard_count: 1,
            tensor_entries: 2,
            selected_source_bytes: 14,
            packed_source_bytes: 14,
        };
        let selection = PackedSelectionRow {
            linear_id: "w".to_string(),
            descriptor: PackedWeight::try_new(
                WeightFormat::E4m3Block128 {
                    scale: poot_quant::format::ScaleEncoding::F32,
                },
                [2, 5],
            )
            .unwrap(),
            weight_name: "w".to_string(),
            scale_name: "s".to_string(),
            shard: SHARD.to_string(),
            weight_span: SourceSpan::new(0, 10),
            scale_span: SourceSpan::new(10, 14),
            weight_dtype: "F8_E4M3".to_string(),
            scale_dtype: "F32".to_string(),
            weight_shape: [2, 5],
            scale_shape: [1, 1],
        };
        let authenticate = |root: &TempRoot| {
            AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
                ArtifactLabel {
                    repository: "local/short-reads".to_string(),
                    revision: "short-reads-revision".to_string(),
                },
                None,
                limits,
                Box::new(CountingReader::new(config.clone(), ReadCount::default())),
                config.clone(),
                Box::new(CountingReader::new(index.clone(), ReadCount::default())),
                BTreeMap::from([(
                    SHARD.to_string(),
                    (None, map_shard(&root.0, SHARD).expect("the shard maps")),
                )]),
            )
            .expect("the unpinned shard authenticates")
        };

        let untouched = TempRoot::new("untouched-shard");
        untouched.write(SHARD, &shard);
        let loaded = authenticate(&untouched)
            .load(
                std::slice::from_ref(&selection),
                &mut PackedOwnerCache::new(),
            )
            .expect("the untouched shard loads");
        assert_eq!(loaded.rows.len(), 1);

        let changed = TempRoot::new("truncated-shard");
        changed.write(SHARD, &shard);
        let mut cache = PackedOwnerCache::new();
        let error = authenticate(&changed)
            .load_with_staging_observer(std::slice::from_ref(&selection), &mut cache, |_| {
                changed.truncate(SHARD);
            })
            .expect_err("a shard truncated after its spans were viewed must refuse the load");
        assert!(
            matches!(error, PackedSafetensorsError::ArtifactChanged { .. }),
            "{error:?}"
        );
        assert!(cache.is_empty());
    }
}
