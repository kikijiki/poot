//! Validation staging, atomic publication, and mixed-source byte accounting.

use super::{
    Arc, AuthenticatedArtifactIdentity, AuthenticatedTensorDescriptor, BTreeMap,
    ClassifiedInventoryRow, ExactSourceBuildCounts, ExactSourceCacheKey, ExactSourceKind,
    ExactSourceOwner, ExactSourceOwnerCache, HASH_BUFFER_BYTES, LoadedPackedLinear,
    MixedCategoryBytes, MixedLoadReport, MixedLoadResult, MixedSourceCategory, OperandRole,
    OwnerCacheKind, PackedBufferKind, PackedLoadReport, PackedLoadResult, PackedOwnerCache,
    PackedOwnerCacheKey, PackedPayload, PackedSafetensorsError, PackedSelectionField,
    PackedSelectionRow, Sha256Digest, SourceRole, TensorDisposition, WeightFormat,
    build_mixed_store, build_range_checks, checked_add,
};

#[derive(Clone)]
pub(crate) struct ValidatedSelection {
    pub(crate) selection: PackedSelectionRow,
    pub(crate) data_start: usize,
}

impl ValidatedSelection {
    pub(crate) fn cache_key(&self, artifact: Sha256Digest) -> PackedOwnerCacheKey {
        PackedOwnerCacheKey {
            artifact,
            descriptor: self.selection.descriptor,
            weight_name: self.selection.weight_name.clone(),
            scale_name: self.selection.scale_name.clone(),
            shard: self.selection.shard.clone(),
            weight_span: self.selection.weight_span,
            scale_span: self.selection.scale_span,
            weight_dtype: self.selection.weight_dtype.clone(),
            scale_dtype: self.selection.scale_dtype.clone(),
            weight_shape: self.selection.weight_shape,
            scale_shape: self.selection.scale_shape,
        }
    }
}

pub(crate) struct StagedRow {
    pub(crate) row: ValidatedSelection,
    pub(crate) key: PackedOwnerCacheKey,
    pub(crate) owner: Arc<PackedPayload>,
    pub(crate) cold: bool,
}

pub(crate) struct PreparedPackedPublication {
    pub(crate) staged: Vec<StagedRow>,
    pub(crate) checks: BTreeMap<String, Vec<RangeCheck>>,
    pub(crate) result: PackedLoadResult,
}

pub(crate) struct PackedWeightDecision {
    pub(crate) ordinal: usize,
    pub(crate) format: WeightFormat,
    pub(crate) logical_shape: [usize; 2],
}

#[derive(Default)]
pub(crate) struct PackedDecisionPair {
    pub(crate) weight: Option<PackedWeightDecision>,
    pub(crate) scale: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct ValidatedExactSelection {
    pub(crate) descriptor: AuthenticatedTensorDescriptor,
    pub(crate) kind: ExactSourceKind,
    pub(crate) data_start: usize,
    pub(crate) key: ExactSourceCacheKey,
}

pub(crate) struct ValidatedMixedSelection {
    pub(crate) packed: Vec<ValidatedSelection>,
    pub(crate) exact: Vec<ValidatedExactSelection>,
    pub(crate) inventory: Vec<ClassifiedInventoryRow>,
    pub(crate) accounting: MixedAccountingTable,
}

pub(crate) struct StagedExactRow {
    pub(crate) row: ValidatedExactSelection,
    pub(crate) owner: Arc<ExactSourceOwner>,
    pub(crate) cold: bool,
}

pub(crate) struct StagedMixedSelection {
    pub(crate) packed: Vec<StagedRow>,
    pub(crate) exact: Vec<StagedExactRow>,
    pub(crate) inventory: Vec<ClassifiedInventoryRow>,
    pub(crate) report: MixedLoadReport,
}

pub(crate) enum StagedOwnerRef<'a> {
    Packed(&'a Arc<PackedPayload>),
    Exact {
        #[cfg(test)]
        read_allocation: *const [u8],
        owner: &'a Arc<ExactSourceOwner>,
    },
}

impl StagedOwnerRef<'_> {
    pub(crate) fn strong_count(&self) -> usize {
        match self {
            Self::Packed(owner) => Arc::strong_count(owner),
            Self::Exact { owner, .. } => Arc::strong_count(owner),
        }
    }

    #[cfg(test)]
    pub(crate) fn preserves_read_allocation(&self) -> bool {
        match self {
            Self::Packed(_) => true,
            Self::Exact {
                read_allocation,
                owner,
            } => std::ptr::eq(*read_allocation, Arc::as_ptr(&owner.bytes)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum MixedByteQuantity {
    Source = 0,
    SelectedSource = 1,
    SelectedRangeRead = 2,
    NewOwner = 3,
    ReusedOwner = 4,
    AggregateOwner = 5,
    PeakInFlightRead = 6,
    PeakUnpublishedOwner = 7,
}

impl MixedByteQuantity {
    pub(crate) const ALL: [Self; 8] = [
        Self::Source,
        Self::SelectedSource,
        Self::SelectedRangeRead,
        Self::NewOwner,
        Self::ReusedOwner,
        Self::AggregateOwner,
        Self::PeakInFlightRead,
        Self::PeakUnpublishedOwner,
    ];

    #[cfg(test)]
    pub(crate) const ADDITIVE: [Self; 6] = [
        Self::Source,
        Self::SelectedSource,
        Self::SelectedRangeRead,
        Self::NewOwner,
        Self::ReusedOwner,
        Self::PeakUnpublishedOwner,
    ];

    pub(crate) const fn field(self) -> &'static str {
        match self {
            Self::Source => "mixed source bytes",
            Self::SelectedSource => "mixed selected source bytes",
            Self::SelectedRangeRead => "mixed selected range-read bytes",
            Self::NewOwner => "mixed new owner bytes",
            Self::ReusedOwner => "mixed reused owner bytes",
            Self::AggregateOwner => "mixed aggregate owner bytes",
            Self::PeakInFlightRead => "mixed peak in-flight read bytes",
            Self::PeakUnpublishedOwner => "mixed unpublished owner bytes",
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct MixedAccountingTable {
    pub(crate) values: [[usize; MixedSourceCategory::ALL.len()]; MixedByteQuantity::ALL.len()],
}

impl MixedAccountingTable {
    pub(crate) fn checked_add(
        &mut self,
        quantity: MixedByteQuantity,
        category: MixedSourceCategory,
        bytes: usize,
    ) -> Result<(), PackedSafetensorsError> {
        checked_add_category(
            &mut self.values[quantity as usize],
            category,
            bytes,
            quantity.field(),
        )
    }

    pub(crate) fn record_peak(
        &mut self,
        quantity: MixedByteQuantity,
        category: MixedSourceCategory,
        bytes: usize,
    ) {
        let value = &mut self.values[quantity as usize][category as usize];
        *value = (*value).max(bytes);
    }

    pub(crate) fn derive_aggregate_owner(&mut self) -> Result<(), PackedSafetensorsError> {
        for category in MixedSourceCategory::ALL {
            self.values[MixedByteQuantity::AggregateOwner as usize][category as usize] =
                checked_add(
                    self.values[MixedByteQuantity::NewOwner as usize][category as usize],
                    self.values[MixedByteQuantity::ReusedOwner as usize][category as usize],
                    MixedByteQuantity::AggregateOwner.field(),
                )?;
        }
        Ok(())
    }

    pub(crate) fn summarize(
        &self,
        quantity: MixedByteQuantity,
    ) -> Result<MixedCategoryBytes, PackedSafetensorsError> {
        let values = self.values[quantity as usize];
        if quantity == MixedByteQuantity::PeakInFlightRead {
            Ok(category_max(values))
        } else {
            category_sum(values, quantity.field())
        }
    }
}

pub(crate) struct PreparedMixedPublication {
    pub(crate) staged: StagedMixedSelection,
    pub(crate) checks: BTreeMap<String, Vec<RangeCheck>>,
    pub(crate) result: MixedLoadResult,
}

pub(crate) struct MixedReportInputs<'a> {
    pub(crate) authentication: PackedLoadReport,
    pub(crate) accounting: MixedAccountingTable,
    pub(crate) packed: &'a [ValidatedSelection],
    pub(crate) exact: &'a [ValidatedExactSelection],
    pub(crate) packed_cache: &'a PackedOwnerCache,
    pub(crate) exact_cache: &'a ExactSourceOwnerCache,
    pub(crate) artifact_cache_identity: Sha256Digest,
    pub(crate) packed_report: PackedLoadReport,
}

pub(crate) fn prepare_packed_publication(
    staged: Vec<StagedRow>,
    report: PackedLoadReport,
    cache: &mut PackedOwnerCache,
) -> Result<PreparedPackedPublication, PackedSafetensorsError> {
    let checks = build_range_checks(&staged)?;
    let cold_count = staged.iter().filter(|row| row.cold).count();
    cache
        .owners
        .try_reserve(cold_count)
        .map_err(|_| PackedSafetensorsError::CacheReservation {
            cache: OwnerCacheKind::Packed,
            additional: cold_count,
        })?;
    let rows = staged
        .iter()
        .map(|staged_row| LoadedPackedLinear {
            linear_id: staged_row.row.selection.linear_id.clone(),
            descriptor: staged_row.row.selection.descriptor,
            weight_name: staged_row.row.selection.weight_name.clone(),
            scale_name: staged_row.row.selection.scale_name.clone(),
            shard: staged_row.row.selection.shard.clone(),
            weight_span: staged_row.row.selection.weight_span,
            scale_span: staged_row.row.selection.scale_span,
            owner: Arc::clone(&staged_row.owner),
        })
        .collect();
    Ok(PreparedPackedPublication {
        staged,
        checks,
        result: PackedLoadResult { rows, report },
    })
}

pub(crate) fn commit_packed_publication(
    mut prepared: PreparedPackedPublication,
    final_verification_bytes: usize,
    cache: &mut PackedOwnerCache,
) -> PackedLoadResult {
    prepared.result.report.final_verification_bytes = final_verification_bytes;
    for staged_row in prepared.staged {
        if staged_row.cold {
            let previous = cache.owners.insert(staged_row.key, staged_row.owner);
            debug_assert!(
                previous.is_none(),
                "a cold packed cache key became occupied"
            );
        }
    }
    prepared.result
}

pub(crate) fn prepare_mixed_publication(
    staged: StagedMixedSelection,
    artifact: AuthenticatedArtifactIdentity,
    packed_cache: &mut PackedOwnerCache,
    exact_cache: &mut ExactSourceOwnerCache,
) -> Result<PreparedMixedPublication, PackedSafetensorsError> {
    let checks = build_mixed_range_checks(&staged.packed, &staged.exact)?;
    let cold_packed = staged.packed.iter().filter(|row| row.cold).count();
    let cold_exact = staged.exact.iter().filter(|row| row.cold).count();
    packed_cache.owners.try_reserve(cold_packed).map_err(|_| {
        PackedSafetensorsError::CacheReservation {
            cache: OwnerCacheKind::Packed,
            additional: cold_packed,
        }
    })?;
    exact_cache.owners.try_reserve(cold_exact).map_err(|_| {
        PackedSafetensorsError::CacheReservation {
            cache: OwnerCacheKind::ExactSource,
            additional: cold_exact,
        }
    })?;
    let (store, packed_metadata, exact_metadata) = build_mixed_store(&staged.packed, &staged.exact)
        .map_err(|error| PackedSafetensorsError::WeightStoreBuild { error })?;
    let mut result = MixedLoadResult {
        artifact,
        inventory: staged.inventory.clone(),
        store,
        packed_metadata,
        exact_metadata,
        report: staged.report,
    };
    result.report.stored_bytes = result.store.total_stored_bytes();
    Ok(PreparedMixedPublication {
        staged,
        checks,
        result,
    })
}

pub(crate) fn commit_mixed_publication(
    mut prepared: PreparedMixedPublication,
    final_hash_io_bytes: usize,
    packed_cache: &mut PackedOwnerCache,
    exact_cache: &mut ExactSourceOwnerCache,
) -> MixedLoadResult {
    prepared.result.report.final_hash_io_bytes = final_hash_io_bytes;
    for row in prepared.staged.packed {
        if row.cold {
            let previous = packed_cache.owners.insert(row.key, row.owner);
            debug_assert!(
                previous.is_none(),
                "a cold packed cache key became occupied"
            );
        }
    }
    for row in prepared.staged.exact {
        if row.cold {
            let previous = exact_cache.owners.insert(row.row.key, row.owner);
            debug_assert!(
                previous.is_none(),
                "a cold exact-source cache key became occupied"
            );
        }
    }
    prepared.result
}

pub(crate) fn build_mixed_report(
    inputs: MixedReportInputs<'_>,
    observe_cache_probe: &mut impl FnMut(OwnerCacheKind),
) -> Result<MixedLoadReport, PackedSafetensorsError> {
    let MixedReportInputs {
        authentication,
        mut accounting,
        packed,
        exact,
        packed_cache,
        exact_cache,
        artifact_cache_identity,
        packed_report,
    } = inputs;
    let mut exact_source_builds = ExactSourceBuildCounts::default();
    let mut packed_payload_builds = 0usize;

    for row in packed {
        let bytes = row.selection.descriptor.total_source_bytes();
        observe_cache_probe(OwnerCacheKind::Packed);
        if !packed_cache
            .owners
            .contains_key(&row.cache_key(artifact_cache_identity))
        {
            for quantity in [
                MixedByteQuantity::SelectedRangeRead,
                MixedByteQuantity::NewOwner,
                MixedByteQuantity::PeakUnpublishedOwner,
            ] {
                accounting.checked_add(quantity, MixedSourceCategory::Packed, bytes)?;
            }
            accounting.record_peak(
                MixedByteQuantity::PeakInFlightRead,
                MixedSourceCategory::Packed,
                bytes,
            );
            checked_increment_packed_builds(&mut packed_payload_builds)?;
        } else {
            accounting.checked_add(
                MixedByteQuantity::ReusedOwner,
                MixedSourceCategory::Packed,
                bytes,
            )?;
        }
    }
    for row in exact {
        let bytes = row.descriptor.span.end - row.descriptor.span.start;
        let category = category_for_exact_kind(row.kind);
        observe_cache_probe(OwnerCacheKind::ExactSource);
        if !exact_cache.owners.contains_key(&row.key) {
            for quantity in [
                MixedByteQuantity::SelectedRangeRead,
                MixedByteQuantity::NewOwner,
                MixedByteQuantity::PeakUnpublishedOwner,
            ] {
                accounting.checked_add(quantity, category, bytes)?;
            }
            accounting.record_peak(MixedByteQuantity::PeakInFlightRead, category, bytes);
            checked_increment_exact_builds(&mut exact_source_builds, row.kind)?;
        } else {
            accounting.checked_add(MixedByteQuantity::ReusedOwner, category, bytes)?;
        }
    }
    accounting.derive_aggregate_owner()?;
    let metadata_bytes = mixed_metadata_bytes(authentication)?;
    exact_source_builds.total = checked_sum(
        &exact_source_builds.values,
        "exact-source build count total",
    )?;
    Ok(MixedLoadReport {
        source_bytes: accounting.summarize(MixedByteQuantity::Source)?,
        selected_source_bytes: accounting.summarize(MixedByteQuantity::SelectedSource)?,
        selected_range_read_bytes: accounting.summarize(MixedByteQuantity::SelectedRangeRead)?,
        new_owner_bytes: accounting.summarize(MixedByteQuantity::NewOwner)?,
        reused_owner_bytes: accounting.summarize(MixedByteQuantity::ReusedOwner)?,
        aggregate_owner_bytes: accounting.summarize(MixedByteQuantity::AggregateOwner)?,
        peak_in_flight_read_bytes: accounting.summarize(MixedByteQuantity::PeakInFlightRead)?,
        peak_unpublished_owner_bytes: accounting
            .summarize(MixedByteQuantity::PeakUnpublishedOwner)?,
        config_bytes: authentication.config_bytes,
        index_bytes: authentication.index_bytes,
        header_bytes_total: authentication.header_bytes_total,
        header_bytes_max: authentication.header_bytes_max,
        metadata_bytes,
        initial_hash_io_bytes: authentication.initial_artifact_hash_bytes,
        final_hash_io_bytes: 0,
        hash_buffer_bytes: HASH_BUFFER_BYTES,
        packed_weight_source_bytes: packed_report.selected_weight_source_bytes,
        packed_scale_source_bytes: packed_report.selected_scale_source_bytes,
        packed_source_padding_bits: packed_report.source_padding_bits,
        forbidden_f32_weight_bytes: packed_report.forbidden_f32_weight_bytes,
        packed_payload_builds,
        // Filled by `prepare_mixed_publication` once `exact_sources`/`packed` (and so the weight
        // store) exist; this function only sees pre-publication row descriptors.
        stored_bytes: 0,
        exact_source_builds,
    })
}

pub(crate) fn checked_increment_packed_builds(
    count: &mut usize,
) -> Result<(), PackedSafetensorsError> {
    *count = checked_add(*count, 1, "mixed packed payload build count")?;
    Ok(())
}

pub(crate) fn checked_increment_exact_builds(
    counts: &mut ExactSourceBuildCounts,
    kind: ExactSourceKind,
) -> Result<(), PackedSafetensorsError> {
    counts.values[kind as usize] =
        checked_add(counts.values[kind as usize], 1, "exact-source build count")?;
    Ok(())
}

pub(crate) fn mixed_metadata_bytes(
    authentication: PackedLoadReport,
) -> Result<usize, PackedSafetensorsError> {
    checked_add(
        checked_add(
            authentication.config_bytes,
            authentication.index_bytes,
            "mixed metadata bytes",
        )?,
        authentication.header_bytes_total,
        "mixed metadata bytes",
    )
}

pub(crate) fn build_mixed_range_checks(
    packed: &[StagedRow],
    exact: &[StagedExactRow],
) -> Result<BTreeMap<String, Vec<RangeCheck>>, PackedSafetensorsError> {
    let mut checks = build_range_checks(packed)?;
    for row in exact.iter().filter(|row| row.cold) {
        checks
            .entry(row.row.descriptor.shard.clone())
            .or_default()
            .push(RangeCheck {
                start: checked_add(
                    row.row.data_start,
                    row.row.descriptor.span.start,
                    "exact-source verification start",
                )?,
                owner: RangeCheckOwner::Exact(Arc::clone(&row.owner)),
            });
    }
    for shard_checks in checks.values_mut() {
        shard_checks.sort_unstable_by_key(|check| check.start);
    }
    Ok(checks)
}

pub(crate) fn category_for_disposition(disposition: &TensorDisposition) -> MixedSourceCategory {
    match disposition {
        TensorDisposition::PackedWeight { .. } | TensorDisposition::PackedScale { .. } => {
            MixedSourceCategory::Packed
        }
        TensorDisposition::StandaloneE4m3 => MixedSourceCategory::StandaloneE4m3,
        TensorDisposition::DenseBf16 => MixedSourceCategory::DenseBf16,
        TensorDisposition::DenseF32 => MixedSourceCategory::DenseF32,
        TensorDisposition::DenseI64 => MixedSourceCategory::DenseI64,
        TensorDisposition::Deferred | TensorDisposition::Excluded => {
            MixedSourceCategory::DeferredOrExcluded
        }
    }
}

pub(crate) fn disposition_selects_source(disposition: &TensorDisposition) -> bool {
    !matches!(
        disposition,
        TensorDisposition::Deferred | TensorDisposition::Excluded
    )
}

pub(crate) fn exact_kind_for_disposition(
    disposition: &TensorDisposition,
) -> Option<ExactSourceKind> {
    match disposition {
        TensorDisposition::StandaloneE4m3 => Some(ExactSourceKind::E4m3),
        TensorDisposition::DenseBf16 => Some(ExactSourceKind::Bf16),
        TensorDisposition::DenseF32 => Some(ExactSourceKind::F32),
        TensorDisposition::DenseI64 => Some(ExactSourceKind::I64),
        _ => None,
    }
}

pub(crate) const fn category_for_exact_kind(kind: ExactSourceKind) -> MixedSourceCategory {
    match kind {
        ExactSourceKind::E4m3 => MixedSourceCategory::StandaloneE4m3,
        ExactSourceKind::Bf16 => MixedSourceCategory::DenseBf16,
        ExactSourceKind::F32 => MixedSourceCategory::DenseF32,
        ExactSourceKind::I64 => MixedSourceCategory::DenseI64,
    }
}

pub(crate) fn checked_add_category(
    values: &mut [usize; MixedSourceCategory::ALL.len()],
    category: MixedSourceCategory,
    bytes: usize,
    field: &'static str,
) -> Result<(), PackedSafetensorsError> {
    values[category as usize] = checked_add(values[category as usize], bytes, field)?;
    Ok(())
}

pub(crate) fn checked_sum(
    values: &[usize],
    field: &'static str,
) -> Result<usize, PackedSafetensorsError> {
    values
        .iter()
        .try_fold(0usize, |total, value| checked_add(total, *value, field))
}

pub(crate) fn category_sum(
    values: [usize; MixedSourceCategory::ALL.len()],
    field: &'static str,
) -> Result<MixedCategoryBytes, PackedSafetensorsError> {
    Ok(MixedCategoryBytes {
        aggregate: checked_sum(&values, field)?,
        values,
    })
}

pub(crate) fn category_max(values: [usize; MixedSourceCategory::ALL.len()]) -> MixedCategoryBytes {
    MixedCategoryBytes {
        aggregate: values.into_iter().max().unwrap_or(0),
        values,
    }
}

pub(crate) fn shape_2(
    linear_id: &str,
    field: PackedSelectionField,
    descriptor: &AuthenticatedTensorDescriptor,
) -> Result<[usize; 2], PackedSafetensorsError> {
    descriptor
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| PackedSafetensorsError::SelectionMismatch {
            linear_id: linear_id.to_string(),
            field,
            expected: "rank-two physical shape".to_string(),
            actual: format!("{:?}", descriptor.shape),
        })
}

pub(crate) struct RangeCheck {
    pub(crate) start: usize,
    pub(crate) owner: RangeCheckOwner,
}

pub(crate) enum RangeCheckOwner {
    Packed {
        owner: Arc<PackedPayload>,
        buffer: PackedBufferKind,
    },
    Exact(Arc<ExactSourceOwner>),
}

impl RangeCheck {
    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.owner {
            RangeCheckOwner::Packed {
                owner,
                buffer: PackedBufferKind::Weight,
            } => owner.bytes(SourceRole::Planar(OperandRole::Codes)),
            RangeCheckOwner::Packed {
                owner,
                buffer: PackedBufferKind::Scale,
            } => owner.bytes(SourceRole::Planar(OperandRole::Scale)),
            RangeCheckOwner::Exact(owner) => owner.bytes(),
        }
    }
}
