//! Builds a [`poot_quant::weights::WeightStore`] from one mixed snapshot's staged rows: the one
//! load output every safetensors reader produces (card 540a). Called once, from
//! [`prepare_mixed_publication`](super::prepare_mixed_publication) while it still holds the
//! staged packed and exact rows, so the store is [`MixedLoadResult`]'s own field from
//! construction - never a view built from that result's fields, counted, and dropped.
//!
//! The mixed loader already reads every tensor it touches as raw bytes - a dense entry
//! ([`ExactSourceOwner`]) or a packed one ([`PackedPayload`], staged as a [`StagedRow`]) - so
//! building the store here is a pure re-keying of data this module already holds; no additional
//! file I/O.

use poot_tensor::DType;
use std::sync::Arc;

use poot_quant::weights::{DenseWeight, DenseWeightError, WeightEntry, WeightStore};

use super::{
    BTreeMap, ExactSourceMetadata, ExactSourceOwner, PackedSourceMetadata, StagedExactRow,
    StagedRow,
};

/// Typed failures building a [`WeightStore`] from one mixed snapshot's staged rows. The mixed
/// loader already authenticated and range-checked every tensor, so these fire only on a loader
/// bug (a byte count or a name that does not match what the loader itself just published).
#[derive(Debug, thiserror::Error)]
pub enum MixedWeightStoreError {
    /// An exact source's authenticated dtype string names no [`DType`].
    #[error("{name}: authenticated dtype {dtype:?} is not a DType")]
    UnknownDtype { name: String, dtype: &'static str },
    /// An exact source's authenticated byte count does not match its dtype and shape.
    #[error("{name}: {error}")]
    Dense {
        name: String,
        #[source]
        error: DenseWeightError,
    },
    /// Two rows (an exact source and a packed linear, or two of either) published the same key.
    #[error("two published rows share the weight key {key:?}")]
    DuplicateKey { key: String },
}

fn dense_entry(owner: &Arc<ExactSourceOwner>) -> Result<WeightEntry, MixedWeightStoreError> {
    let name = owner.descriptor().name();
    let dtype_str = owner.kind().dtype();
    let dtype = DType::parse(dtype_str).map_err(|_| MixedWeightStoreError::UnknownDtype {
        name: name.to_string(),
        dtype: dtype_str,
    })?;
    let dense = DenseWeight::try_new(
        dtype,
        owner.descriptor().shape().to_vec(),
        Arc::clone(&owner.bytes),
    )
    .map_err(|error| MixedWeightStoreError::Dense {
        name: name.to_string(),
        error,
    })?;
    Ok(WeightEntry::Dense(dense))
}

/// [`build_mixed_store`]'s output: the [`WeightStore`] itself plus the checkpoint provenance it
/// cannot express, one map per entry kind, each keyed the same way as the store (linear id for a
/// packed row, tensor name for an exact/dense one).
pub(crate) type MixedStoreParts = (
    WeightStore,
    BTreeMap<String, PackedSourceMetadata>,
    BTreeMap<String, ExactSourceMetadata>,
);

/// Builds this snapshot's [`WeightStore`] - one [`WeightEntry::Dense`] per `staged_exact` row
/// (keyed by its authenticated tensor name) and one [`WeightEntry::Packed`] per `staged_packed`
/// row (keyed by its linear id) - plus the checkpoint provenance the store cannot express, one map
/// per kind, keyed the same way as the store.
pub(crate) fn build_mixed_store(
    staged_packed: &[StagedRow],
    staged_exact: &[StagedExactRow],
) -> Result<MixedStoreParts, MixedWeightStoreError> {
    let mut builder = WeightStore::builder();
    let mut packed_metadata = BTreeMap::new();
    for staged in staged_packed {
        let selection = &staged.row.selection;
        builder
            .insert(
                selection.linear_id.clone(),
                WeightEntry::Packed(Arc::clone(&staged.owner)),
            )
            .map_err(|error| MixedWeightStoreError::DuplicateKey {
                key: error.0.to_string(),
            })?;
        packed_metadata.insert(
            selection.linear_id.clone(),
            PackedSourceMetadata {
                weight_name: selection.weight_name.clone(),
                scale_name: selection.scale_name.clone(),
                shard: selection.shard.clone(),
                weight_span: selection.weight_span,
                scale_span: selection.scale_span,
            },
        );
    }
    let mut exact_metadata = BTreeMap::new();
    for staged in staged_exact {
        let name = staged.row.descriptor.name().to_string();
        builder
            .insert(name.clone(), dense_entry(&staged.owner)?)
            .map_err(|error| MixedWeightStoreError::DuplicateKey {
                key: error.0.to_string(),
            })?;
        exact_metadata.insert(
            name,
            ExactSourceMetadata {
                kind: staged.row.kind,
                shard: staged.row.descriptor.shard().to_string(),
                span: staged.row.descriptor.span(),
                owner: Arc::clone(&staged.owner),
            },
        );
    }
    Ok((builder.build(), packed_metadata, exact_metadata))
}

#[cfg(test)]
mod tests {
    use poot_quant::format::WeightFormat;
    use poot_quant::{PackedPayload, PackedWeight, SourceRole};

    use super::*;
    use crate::packed_safetensors::{
        ArtifactIdentityFields, AuthenticatedArtifactIdentity, AuthenticatedTensorDescriptor,
        ExactSourceCacheKey, ExactSourceKind, PackedSelectionRow, Sha256Digest, SourceSpan,
        ValidatedExactSelection, ValidatedSelection,
    };

    fn artifact() -> AuthenticatedArtifactIdentity {
        AuthenticatedArtifactIdentity(Arc::new(ArtifactIdentityFields {
            repository: "synthetic/repo".to_string(),
            revision: "main".to_string(),
            config_length: 0,
            config_sha256: Sha256Digest([0u8; 32]),
            index_length: 0,
            index_sha256: Sha256Digest([0u8; 32]),
            shards: Vec::new(),
        }))
    }

    fn descriptor(
        name: &str,
        dtype: &str,
        shape: Vec<usize>,
        len: usize,
    ) -> AuthenticatedTensorDescriptor {
        AuthenticatedTensorDescriptor {
            name: name.to_string(),
            shard: "shard.safetensors".to_string(),
            span: SourceSpan::new(0, len),
            dtype: dtype.to_string(),
            shape,
        }
    }

    fn staged_exact_row(
        descriptor: AuthenticatedTensorDescriptor,
        kind: ExactSourceKind,
        bytes: Vec<u8>,
    ) -> StagedExactRow {
        let owner = Arc::new(ExactSourceOwner {
            artifact: artifact(),
            descriptor: descriptor.clone(),
            kind,
            bytes: Arc::from(bytes),
        });
        StagedExactRow {
            row: ValidatedExactSelection {
                descriptor,
                kind,
                data_start: 0,
                key: ExactSourceCacheKey {
                    artifact: artifact(),
                    descriptor: owner.descriptor.clone(),
                    kind,
                },
            },
            owner,
            cold: true,
        }
    }

    fn staged_packed_row(linear_id: &str, owner: Arc<PackedPayload>) -> StagedRow {
        let weight = owner.weight();
        let block_bytes = weight.source_bytes(SourceRole::Blocks);
        let selection = PackedSelectionRow {
            linear_id: linear_id.to_string(),
            descriptor: weight,
            weight_name: format!("{linear_id}.weight"),
            scale_name: String::new(),
            shard: "shard.safetensors".to_string(),
            weight_span: SourceSpan::new(0, block_bytes),
            scale_span: SourceSpan::new(0, 0),
            weight_dtype: "Q4_0".to_string(),
            scale_dtype: String::new(),
            weight_shape: weight.shape(),
            scale_shape: [0, 0],
        };
        let row = ValidatedSelection {
            selection,
            data_start: 0,
        };
        let key = row.cache_key(artifact_digest());
        StagedRow {
            row,
            key,
            owner,
            cold: true,
        }
    }

    fn artifact_digest() -> Sha256Digest {
        Sha256Digest([0u8; 32])
    }

    #[test]
    fn weight_store_holds_one_dense_entry_per_exact_source() {
        let row = staged_exact_row(
            descriptor("embed_tokens.weight", "F32", vec![2], 8),
            ExactSourceKind::F32,
            vec![0u8; 8],
        );

        let (store, packed_metadata, exact_metadata) =
            build_mixed_store(&[], std::slice::from_ref(&row)).unwrap();
        assert_eq!(store.len(), 1);
        assert!(packed_metadata.is_empty());
        let entry = store.get("embed_tokens.weight").unwrap();
        assert_eq!(entry.shape(), vec![2]);
        assert_eq!(entry.stored_bytes(), 8);
        assert_eq!(store.total_stored_bytes(), 8);
        let meta = exact_metadata.get("embed_tokens.weight").unwrap();
        assert_eq!(meta.kind, ExactSourceKind::F32);
        assert_eq!(meta.shard, "shard.safetensors");
        assert!(Arc::ptr_eq(&meta.owner, &row.owner));
    }

    #[test]
    fn weight_store_accepts_standalone_e4m3_and_i64_exact_sources() {
        // A standalone (unpaired) F8_E4M3 tensor and an I64 index buffer are dense
        // stored tensors; the store must hold both, not reject or drop them.
        let e4m3 = staged_exact_row(
            descriptor("layers.0.mlp.gate.weight", "F8_E4M3", vec![4, 4], 16),
            ExactSourceKind::E4m3,
            vec![0u8; 16],
        );
        let indices = staged_exact_row(
            descriptor("embeddings.position_ids", "I64", vec![3], 24),
            ExactSourceKind::I64,
            vec![0u8; 24],
        );

        let (store, _, exact_metadata) = build_mixed_store(&[], &[e4m3, indices]).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(exact_metadata.len(), 2);
        assert_eq!(
            store
                .get("layers.0.mlp.gate.weight")
                .unwrap()
                .stored_bytes(),
            16
        );
        assert_eq!(
            store.get("embeddings.position_ids").unwrap().stored_bytes(),
            24
        );
        assert_eq!(store.total_stored_bytes(), 16 + 24);
    }

    #[test]
    fn weight_store_holds_one_packed_entry_per_linear_keyed_by_linear_id() {
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, [1, 32]).unwrap();
        let block_bytes = weight.source_bytes(SourceRole::Blocks);
        let owner = Arc::new(
            PackedPayload::try_new(
                weight,
                [(SourceRole::Blocks, Arc::from(vec![0u8; block_bytes]))],
            )
            .unwrap(),
        );
        let row = staged_packed_row("layers.0.mlp.down_proj", owner);

        let (store, packed_metadata, exact_metadata) =
            build_mixed_store(std::slice::from_ref(&row), &[]).unwrap();
        assert_eq!(store.len(), 1);
        assert!(exact_metadata.is_empty());
        let entry = store.get("layers.0.mlp.down_proj").unwrap();
        assert_eq!(entry.shape(), vec![1, 32]);
        assert_eq!(entry.stored_bytes(), block_bytes);
        assert_eq!(store.total_stored_bytes(), block_bytes);
        let meta = packed_metadata.get("layers.0.mlp.down_proj").unwrap();
        assert_eq!(meta.weight_name, "layers.0.mlp.down_proj.weight");
        assert_eq!(meta.shard, "shard.safetensors");
        assert_eq!(meta.weight_span, SourceSpan::new(0, block_bytes));
    }

    #[test]
    fn weight_store_rejects_a_byte_count_that_does_not_match_the_authenticated_dtype_and_shape() {
        // Guard for SC-002: an exact source whose bytes disagree with its
        // authenticated shape/dtype must fail typed, not silently publish a truncated tensor.
        let row = staged_exact_row(
            descriptor("broken.weight", "F32", vec![4], 16),
            ExactSourceKind::F32,
            vec![0u8; 8], // half the bytes a [4] F32 tensor needs
        );
        assert!(matches!(
            build_mixed_store(&[], &[row]),
            Err(MixedWeightStoreError::Dense { .. })
        ));
    }

    /// Mutation proof: if `build_mixed_store` ever stopped feeding
    /// `WeightEntry::Packed` from the exact same `Arc<PackedPayload>` the caller staged (e.g. a
    /// change that rebuilt a fresh payload from re-read bytes instead), this pointer-equality
    /// check goes red even though every byte still matches. Verified red by swapping
    /// `Arc::clone(&staged.owner)` for a freshly-built payload with identical content.
    #[test]
    fn packed_entry_is_pointer_equal_to_the_staged_owner() {
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, [1, 32]).unwrap();
        let block_bytes = weight.source_bytes(SourceRole::Blocks);
        let owner = Arc::new(
            PackedPayload::try_new(
                weight,
                [(SourceRole::Blocks, Arc::from(vec![0u8; block_bytes]))],
            )
            .unwrap(),
        );
        let row = staged_packed_row("layers.0.mlp.down_proj", Arc::clone(&owner));

        let (store, ..) = build_mixed_store(std::slice::from_ref(&row), &[]).unwrap();
        match store.get("layers.0.mlp.down_proj").unwrap() {
            WeightEntry::Packed(stored) => assert!(Arc::ptr_eq(stored, &owner)),
            WeightEntry::Dense(_) => panic!("expected a packed entry"),
        }
    }
}
