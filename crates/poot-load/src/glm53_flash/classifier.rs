use super::*;

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct Glm53FlashExpectedShard {
    pub(crate) name: String,
    pub(crate) size: usize,
    pub(crate) sha256: String,
    pub(crate) header_length: usize,
    pub(crate) header_sha256: String,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct Glm53FlashTextArtifactContract {
    pub(crate) repository: String,
    pub(crate) revision: String,
    pub(crate) config_sha256: String,
    pub(crate) index_sha256: String,
    pub(crate) shards: Vec<Glm53FlashExpectedShard>,
}

#[cfg(test)]
pub(crate) fn validate_glm53_flash_text_artifact(
    artifact: &AuthenticatedArtifactIdentity,
    expected: &Glm53FlashTextArtifactContract,
) -> Result<(), Glm53FlashMetadataError> {
    if artifact.revision() != expected.revision {
        return Err(Glm53FlashMetadataError::TextRevision {
            expected: expected.revision.clone(),
            actual: artifact.revision().to_string(),
        });
    }
    for (field, expected, actual) in [
        (
            "repository",
            expected.repository.clone(),
            artifact.repository().to_string(),
        ),
        (
            "config SHA-256",
            expected.config_sha256.clone(),
            sha256_digest_hex(artifact.config_sha256()),
        ),
        (
            "index SHA-256",
            expected.index_sha256.clone(),
            sha256_digest_hex(artifact.index_sha256()),
        ),
    ] {
        if actual != expected {
            return Err(Glm53FlashMetadataError::TextArtifact {
                field,
                expected,
                actual,
            });
        }
    }
    if artifact.shards().len() != expected.shards.len() {
        return Err(Glm53FlashMetadataError::TextArtifact {
            field: "shard count",
            expected: expected.shards.len().to_string(),
            actual: artifact.shards().len().to_string(),
        });
    }
    for expected in &expected.shards {
        let actual = artifact
            .shards()
            .iter()
            .find(|shard| shard.filename == expected.name)
            .ok_or_else(|| Glm53FlashMetadataError::TextArtifact {
                field: "shard name",
                expected: expected.name.clone(),
                actual: "missing".to_string(),
            })?;
        for (field, expected_value, actual_value) in [
            (
                "shard length",
                expected.size.to_string(),
                actual.file_length.to_string(),
            ),
            (
                "shard SHA-256",
                expected.sha256.clone(),
                sha256_digest_hex(actual.file_sha256),
            ),
            (
                "shard header length",
                expected.header_length.to_string(),
                actual.header_length.to_string(),
            ),
            (
                "shard header SHA-256",
                expected.header_sha256.clone(),
                sha256_digest_hex(actual.header_sha256),
            ),
        ] {
            if actual_value != expected_value {
                return Err(Glm53FlashMetadataError::TextArtifact {
                    field,
                    expected: format!("{} for {}", expected_value, expected.name),
                    actual: actual_value,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_glm53_flash_text_row(
    expected: &Glm53FlashExpectedTextRow,
    actual_name: &str,
    actual_shard: &str,
    actual_dtype: &str,
    actual_shape: &[usize],
) -> Result<(), Glm53FlashMetadataError> {
    let mut fields = vec![
        ("name", expected.name.clone(), actual_name.to_string()),
        ("shard", expected.shard.clone(), actual_shard.to_string()),
    ];
    if let Glm53FlashExpectedTextKind::Selected { dtype, shape, .. } = &expected.kind {
        fields.push(("dtype", dtype.clone(), actual_dtype.to_string()));
        fields.push(("shape", format!("{shape:?}"), format!("{actual_shape:?}")));
    }
    for (field, expected_value, actual_value) in fields {
        if actual_value != expected_value {
            return Err(Glm53FlashMetadataError::TextInventoryRow {
                name: expected.name.clone(),
                field,
                expected: expected_value,
                actual: actual_value,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct Glm53FlashTextContract {
    pub(crate) artifact: Glm53FlashTextArtifactContract,
    pub(crate) rows: Vec<Glm53FlashExpectedTextRow>,
    pub(crate) report: Glm53FlashTextInventoryReport,
}

/// Sealed, digest-bound GLM-5.3-Flash text classification contract.
///
/// The fields are private so callers cannot pair an identity from one Card 359 authentication with an
/// inventory from another. [`Glm53FlashTextClassifier::load`] obtains both from one held transaction.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct Glm53FlashTextClassifier {
    pub(crate) contract: Glm53FlashTextContract,
}

/// Authenticated Card 359 owners plus the classifier's exact partition report.
#[derive(Clone, Debug)]
pub struct Glm53FlashTextLoadResult {
    pub(crate) mixed: MixedLoadResult,
    pub(crate) report: Glm53FlashTextInventoryReport,
}

impl Glm53FlashTextLoadResult {
    pub const fn mixed(&self) -> &MixedLoadResult {
        &self.mixed
    }

    pub const fn report(&self) -> Glm53FlashTextInventoryReport {
        self.report
    }
}

#[cfg(test)]
#[derive(Debug, thiserror::Error)]
pub(crate) enum Glm53FlashTextLoadError {
    #[error("GLM-5.3-Flash classifier rejected the authenticated artifact: {0}")]
    Artifact(#[source] Glm53FlashMetadataError),
    #[error(transparent)]
    Transaction(#[from] MixedLoadError<Glm53FlashMetadataError>),
}

#[cfg(test)]
pub(crate) fn classify_glm53_flash_text_inventory(
    rows: &[Glm53FlashExpectedTextRow],
    inventory: AuthenticatedInventory<'_>,
) -> Result<Vec<InventoryDecision>, Glm53FlashMetadataError> {
    if inventory.len() != rows.len() {
        return Err(Glm53FlashMetadataError::TextInventoryCount {
            expected: rows.len(),
            actual: inventory.len(),
        });
    }
    let mut decisions = Vec::with_capacity(inventory.len());
    for (expected, actual) in rows.iter().zip(inventory.rows()) {
        validate_glm53_flash_text_row(
            expected,
            actual.name(),
            actual.shard(),
            actual.dtype(),
            actual.shape(),
        )?;
        decisions.push(InventoryDecision::new(
            actual.key(),
            glm53_flash_text_disposition(expected)?,
        ));
    }
    Ok(decisions)
}

#[cfg(test)]
impl Glm53FlashTextClassifier {
    pub(crate) const fn report(&self) -> Glm53FlashTextInventoryReport {
        self.contract.report
    }

    /// Classify and load owners inside one authenticated Card 359 transaction.
    pub(crate) fn load(
        &self,
        authenticated: &mut AuthenticatedSafetensorsHandleSet,
        packed_cache: &mut PackedOwnerCache,
        exact_cache: &mut ExactSourceOwnerCache,
    ) -> Result<Glm53FlashTextLoadResult, Glm53FlashTextLoadError> {
        self.load_with_decision_mutation(authenticated, packed_cache, exact_cache, |_| {})
    }

    pub(crate) fn load_with_decision_mutation(
        &self,
        authenticated: &mut AuthenticatedSafetensorsHandleSet,
        packed_cache: &mut PackedOwnerCache,
        exact_cache: &mut ExactSourceOwnerCache,
        mutate: impl FnOnce(&mut Vec<InventoryDecision>),
    ) -> Result<Glm53FlashTextLoadResult, Glm53FlashTextLoadError> {
        validate_glm53_flash_text_artifact(
            authenticated.artifact_identity(),
            &self.contract.artifact,
        )
        .map_err(Glm53FlashTextLoadError::Artifact)?;
        let rows = &self.contract.rows;
        let mixed = authenticated.load_mixed(packed_cache, exact_cache, |inventory| {
            let mut decisions = classify_glm53_flash_text_inventory(rows, inventory)?;
            mutate(&mut decisions);
            Ok(decisions)
        })?;
        Ok(Glm53FlashTextLoadResult {
            mixed,
            report: self.contract.report,
        })
    }
}
