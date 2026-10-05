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
use poot_quant::{OperandRole, PackedPayload, PackedWeight, PackedWeightError, SourceRole};

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
pub use validation::FilePin;
pub(crate) use validation::*;

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
        fail_at: Option<Rc<Cell<Option<u64>>>>,
    }

    impl CountingReader {
        fn new(bytes: Vec<u8>, read_count: ReadCount) -> Self {
            Self {
                bytes: Cursor::new(bytes),
                read_count,
                fail_at: None,
            }
        }

        fn with_failure_switch(
            bytes: Vec<u8>,
            read_count: ReadCount,
            fail_at: Rc<Cell<Option<u64>>>,
        ) -> Self {
            Self {
                bytes: Cursor::new(bytes),
                read_count,
                fail_at: Some(fail_at),
            }
        }
    }

    impl Read for CountingReader {
        fn read(&mut self, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
            if self
                .fail_at
                .as_ref()
                .and_then(|fail_at| fail_at.get())
                .is_some_and(|fail_at| self.bytes.position() >= fail_at)
            {
                return Err(std::io::Error::other("injected retained-reader failure"));
            }
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
        shard: ReadCount,
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
                shard: ReadCount::default(),
            };
            let opened_shards = BTreeMap::from([(
                SHARD.to_string(),
                (
                    Some(shard_manifest),
                    Box::new(CountingReader::new(self.shard.clone(), reads.shard.clone()))
                        as RetainedHandle,
                ),
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

    fn assert_reads(reads: &FixtureReads, config: usize, index: usize, shard: usize) {
        assert_eq!(reads.config.get(), config, "config bytes read");
        assert_eq!(reads.index.get(), index, "index bytes read");
        assert_eq!(reads.shard.get(), shard, "shard bytes read");
    }

    #[test]
    fn authentication_retains_the_verified_config_bytes() {
        let fixture = CountingFixture::new();
        let (authenticated, reads) = fixture.authenticate(fixture.manifest.clone(), fixture.limits);
        let authenticated = authenticated.expect("fixture authenticates");

        assert_eq!(authenticated.config_bytes(), fixture.config);
        let header_length = u64::from_le_bytes(fixture.shard[..8].try_into().unwrap()) as usize;
        // Card 544: authentication no longer hashes the shard's full content up front (that pass
        // moves to `verify_before_publication`, once); only the header prefix is read here.
        assert_reads(
            &reads,
            fixture.config.len(),
            fixture.index.len(),
            8 + header_length,
        );
    }

    #[test]
    fn authentication_retains_the_supplied_config_owner_without_rereading_it() {
        let fixture = CountingFixture::new();
        let shard_manifest = fixture.manifest.shards[0].clone();
        let config = fixture.config.clone();
        let config_owner = config.as_ptr();
        let config_reads = ReadCount::default();
        let index_reads = ReadCount::default();
        let shard_reads = ReadCount::default();
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
                    Box::new(CountingReader::new(
                        fixture.shard.clone(),
                        shard_reads.clone(),
                    )) as RetainedHandle,
                ),
            )]),
        )
        .expect("fixture authenticates from its retained config owner");

        assert_eq!(authenticated.config_bytes().as_ptr(), config_owner);
        assert_eq!(config_reads.get(), 0, "config handle must not be reread");
        assert_eq!(index_reads.get(), fixture.index.len());
        let header_length = u64::from_le_bytes(fixture.shard[..8].try_into().unwrap()) as usize;
        // Card 544: only the header prefix is read at authentication time now; the shard's
        // full content streams once, later, at `verify_before_publication`.
        assert_eq!(shard_reads.get(), 8 + header_length);
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

        let opened_shards = BTreeMap::from([(
            SHARD.to_string(),
            (
                None,
                Box::new(CountingReader::new(tampered_shard, ReadCount::default()))
                    as RetainedHandle,
            ),
        )]);
        let mut authenticated =
            AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
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
            .verify_before_publication(&BTreeMap::new())
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
            (
                Some(shard_manifest),
                Box::new(CountingReader::new(tampered_shard, ReadCount::default()))
                    as RetainedHandle,
            ),
        )]);
        let mut authenticated =
            AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
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
            .verify_before_publication(&BTreeMap::new())
            .expect_err("a pinned digest that no longer matches the shard must refuse the load");
        assert!(
            matches!(error, PackedSafetensorsError::ArtifactChanged { .. }),
            "{error:?}"
        );
    }

    /// Card 544 SC-002: a shard streams its full content at most once across a whole load - the
    /// baseline hashed every shard twice (R475-014), once during authentication and again before
    /// publication.
    ///
    /// Mutation: reintroduce the old authentication-time full-shard `hash_file` call ahead of
    /// publication; the second assertion's byte count doubles (header probe plus two full
    /// passes instead of one) and goes red.
    #[test]
    fn card544_sc002_shard_body_streams_at_most_once_across_authenticate_and_publish() {
        let fixture = CountingFixture::new();
        let shard_manifest = fixture.manifest.shards[0].clone();
        let shard_reads = ReadCount::default();
        let opened_shards = BTreeMap::from([(
            SHARD.to_string(),
            (
                Some(shard_manifest),
                Box::new(CountingReader::new(
                    fixture.shard.clone(),
                    shard_reads.clone(),
                )) as RetainedHandle,
            ),
        )]);
        let mut authenticated =
            AuthenticatedSafetensorsHandleSet::authenticate_retained_config_bytes(
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
            .expect("fixture authenticates");

        let header_length = u64::from_le_bytes(fixture.shard[..8].try_into().unwrap()) as usize;
        assert_eq!(
            shard_reads.get(),
            8 + header_length,
            "authentication reads only the header prefix, not the shard's data section"
        );

        authenticated
            .verify_before_publication(&BTreeMap::new())
            .expect("the unmutated shard matches its pin");
        assert_eq!(
            shard_reads.get(),
            8 + header_length + fixture.shard.len(),
            "publication streams the shard's full content exactly once more, not the whole file twice"
        );
    }

    #[derive(Clone, Copy)]
    enum SecondPairFailure {
        Read,
        Build,
    }

    fn rollback_fixture(
        failure: SecondPairFailure,
    ) -> (
        AuthenticatedSafetensorsHandleSet,
        Vec<PackedSelectionRow>,
        Rc<Cell<Option<u64>>>,
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
        let second_weight = match failure {
            SecondPairFailure::Read => 0x02,
            SecondPairFailure::Build => 0x7f,
        };
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
        let data_start = 8 + header.len();
        let shard_manifest = PackedShardManifest {
            filename: SHARD.to_string(),
            file_length: shard.len(),
            file_sha256: sha256_digest(&shard),
            header_length: header.len(),
            header_sha256: sha256_digest(&header),
        };
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
        let fail_at = Rc::new(Cell::new(None));
        let opened_shards = BTreeMap::from([(
            SHARD.to_string(),
            (
                Some(shard_manifest),
                Box::new(CountingReader::with_failure_switch(
                    shard,
                    ReadCount::default(),
                    Rc::clone(&fail_at),
                )) as RetainedHandle,
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
        if matches!(failure, SecondPairFailure::Read) {
            fail_at.set(Some((data_start + 5) as u64));
        }
        (authenticated, selections, fail_at)
    }

    #[test]
    fn second_pair_failure_drops_actual_first_staged_owner() {
        for failure in [SecondPairFailure::Read, SecondPairFailure::Build] {
            let (mut authenticated, selections, _fail_at) = rollback_fixture(failure);
            let mut cache = PackedOwnerCache::new();
            let mut staged_owners = Vec::new();
            let result =
                authenticated.load_with_staging_observer(&selections, &mut cache, |owner| {
                    staged_owners.push(Arc::downgrade(owner))
                });
            match failure {
                SecondPairFailure::Read => assert!(matches!(
                    result,
                    Err(PackedSafetensorsError::SelectedRead {
                        completed_payloads: 1,
                        ..
                    })
                )),
                SecondPairFailure::Build => assert!(matches!(
                    result,
                    Err(PackedSafetensorsError::PayloadBuild {
                        completed_payloads: 1,
                        ..
                    })
                )),
            }
            assert!(cache.is_empty());
            assert_eq!(staged_owners.len(), 1);
            assert!(staged_owners[0].upgrade().is_none());
        }
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

    #[test]
    fn mixed_snapshot_publication_is_atomic() {
        let (mut authenticated, _selections, _fail_at) = rollback_fixture(SecondPairFailure::Read);
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
                StagedOwnerRef::Exact { owner, .. } => exact_owners.push(Arc::downgrade(owner)),
            },
        );
        assert!(matches!(
            result,
            Err(MixedLoadError::Assembly(
                PackedSafetensorsError::ExactSourceRead {
                    completed_owners: 0,
                    ..
                }
            ))
        ));
        assert!(packed_cache.is_empty());
        assert!(exact_cache.is_empty());
        assert_eq!(packed_owners.len(), 1);
        assert!(packed_owners[0].upgrade().is_none());
        assert!(exact_owners.is_empty());
    }

    #[test]
    fn exact_source_read_allocation_is_owner_allocation() {
        let (mut authenticated, _selections, _fail_at) = rollback_fixture(SecondPairFailure::Build);
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let mut handoffs = Vec::new();
        let result = authenticated
            .load_mixed_with_staging_observer(
                &mut packed_cache,
                &mut exact_cache,
                |inventory| Ok::<_, ()>(rollback_mixed_decisions(inventory)),
                |owner| {
                    if let StagedOwnerRef::Exact {
                        read_allocation,
                        owner,
                    } = owner
                    {
                        assert!(std::ptr::eq(read_allocation, Arc::as_ptr(&owner.bytes)));
                        handoffs.push((read_allocation, Arc::downgrade(&owner.bytes)));
                    }
                },
            )
            .unwrap();

        assert_eq!(handoffs.len(), 1);
        let (read_allocation, owner_bytes) = &handoffs[0];
        let owner_bytes = owner_bytes.upgrade().unwrap();
        assert!(std::ptr::eq(*read_allocation, Arc::as_ptr(&owner_bytes)));
        assert_eq!(result.exact_metadata.len(), 1);
        let published = result.exact_metadata.values().next().unwrap();
        assert!(Arc::ptr_eq(&owner_bytes, &published.owner.bytes));
        // Three live `Arc<[u8]>` handles on this allocation: `published.owner.bytes` itself, the
        // `WeightStore`'s own `WeightEntry::Dense` (`Arc::clone`-d from that same owner while
        // `result.store` was built - card 540a: the store is `MixedLoadResult`'s
        // returned field, not a view built and dropped before the caller sees it), and this test's
        // own `owner_bytes` handle.
        assert_eq!(Arc::strong_count(&owner_bytes), 3);
        drop(owner_bytes);
        assert_eq!(Arc::strong_count(&published.owner.bytes), 2);
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
            let (mut authenticated, _selections, _fail_at) =
                rollback_fixture(SecondPairFailure::Build);
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
            let (foreign, _selections, _fail_at) = rollback_fixture(SecondPairFailure::Build);
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
        assert_reads(&reads, 0, 0, 0);

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
        assert_reads(&reads, fixture.config.len(), 0, 0);

        let mut manifest = fixture.manifest.clone();
        manifest.shards[0].file_length += 1;
        let (result, reads) = fixture.authenticate(manifest, fixture.limits);
        assert!(matches!(
            result,
            Err(PackedSafetensorsError::FileLengthMismatch { .. })
        ));
        assert_reads(&reads, fixture.config.len(), fixture.index.len(), 0);

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
        assert_reads(&reads, fixture.config.len(), fixture.index.len(), 8);

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
        assert_reads(&reads, 0, 0, 0);
    }

    /// A shard whose bytes a test can change between the loader's span read and its final
    /// verification pass, served at most `SHORT_READ` bytes per `read`, so that pass sees every
    /// range check in a later read than the first.
    struct ShortReads {
        bytes: Rc<std::cell::RefCell<Vec<u8>>>,
        position: u64,
    }

    const SHORT_READ: usize = 7;

    impl Read for ShortReads {
        fn read(&mut self, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
            let bytes = self.bytes.borrow();
            let start = (self.position as usize).min(bytes.len());
            let read = buffer.len().min(SHORT_READ).min(bytes.len() - start);
            buffer[..read].copy_from_slice(&bytes[start..start + read]);
            self.position += read as u64;
            Ok(read)
        }
    }

    impl Seek for ShortReads {
        fn seek(&mut self, position: SeekFrom) -> Result<u64, std::io::Error> {
            let length = self.bytes.borrow().len() as i64;
            let target = match position {
                SeekFrom::Start(offset) => offset as i64,
                SeekFrom::End(delta) => length + delta,
                SeekFrom::Current(delta) => self.position as i64 + delta,
            };
            self.position = u64::try_from(target)
                .map_err(|_| std::io::Error::other("seek before the start"))?;
            Ok(self.position)
        }
    }

    impl RetainedReader for ShortReads {
        fn file_length(&self) -> Result<u64, std::io::Error> {
            Ok(self.bytes.borrow().len() as u64)
        }
    }

    /// Mutant H4 (mutants-m4.md): `verify_file` compares each staged range with the buffer at
    /// `overlap_start - total`. Made `+ total`, it is right only in the first read (`total == 0`),
    /// and every fixture's range fell inside the first `HASH_BUFFER_BYTES` read. Here the shard
    /// has no pin (so only the range check can see a change) and is served 7 bytes per read. A
    /// weight byte changed after the loader read it must refuse the load, and the untouched shard
    /// must load.
    #[test]
    fn range_check_in_a_later_read_detects_a_changed_byte() {
        let config = br#"{}"#.to_vec();
        let index = format!(r#"{{"weight_map":{{"w":"{SHARD}","s":"{SHARD}"}}}}"#).into_bytes();
        let header = br#"{"w":{"dtype":"F8_E4M3","shape":[2,5],"data_offsets":[0,10]},"s":{"dtype":"F32","shape":[1,1],"data_offsets":[10,14]}}"#;
        let data_start = 8 + header.len();
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
        let authenticate = |bytes: &Rc<std::cell::RefCell<Vec<u8>>>| {
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
                    (
                        None,
                        Box::new(ShortReads {
                            bytes: Rc::clone(bytes),
                            position: 0,
                        }) as RetainedHandle,
                    ),
                )]),
            )
            .expect("the unpinned shard authenticates")
        };

        let untouched = Rc::new(std::cell::RefCell::new(shard.clone()));
        let loaded = authenticate(&untouched)
            .load(
                std::slice::from_ref(&selection),
                &mut PackedOwnerCache::new(),
            )
            .expect("the untouched shard loads");
        assert_eq!(loaded.rows.len(), 1);

        let changed = Rc::new(std::cell::RefCell::new(shard));
        let mut cache = PackedOwnerCache::new();
        let error = authenticate(&changed)
            .load_with_staging_observer(std::slice::from_ref(&selection), &mut cache, |_| {
                changed.borrow_mut()[data_start + 3] ^= 0x01;
            })
            .expect_err("a weight byte changed after its read must refuse the load");
        assert!(
            matches!(error, PackedSafetensorsError::ArtifactChanged { .. }),
            "{error:?}"
        );
        assert!(cache.is_empty());
    }
}
