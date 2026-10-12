//! Placement: one authoritative Card 359 host owner referenced by replicated device residents or exact
//! disjoint byte ranges, never duplicated (FR-002, FR-003).
//!
//! Card 360 does not depend on `poot-load`'s concrete owner types (this crate sits below checkpoint
//! loading in the dependency graph - see `specs/360-packed-multi-device-topology-capture/spec.md`'s prior-
//! art section). [`HostOwner`] carries only the caller-declared identity, logical byte extent, and
//! encoding-unit granularity a Card 359 owner has; it never reads or copies owner bytes.

use std::collections::{HashMap, HashSet};

use crate::multi_device::topology::DeviceId;

/// Opaque identity for one authoritative Card 359 host owner. Two [`PlacementRow`]s that name the same
/// `OwnerId` reference the SAME host payload; Card 360 never constructs a second owner for the same
/// source (FR-002).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwnerId(pub u64);

/// One authoritative Card 359 host owner's logical extent and split granularity. Card 360 carries this
/// identity only; it never inspects or duplicates the owner's authoritative bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostOwner {
    /// The owner's stable identity.
    pub owner: OwnerId,
    /// The owner's whole logical extent, in bytes.
    pub total_bytes: usize,
    /// Bytes per encoding-defined unit (e.g. one packed row); a partition boundary must land on a
    /// multiple of this, or on `total_bytes` itself. `1` means no encoding constraint - not every Card
    /// 359 owner is a packed payload.
    pub encoding_unit_bytes: usize,
}

/// A half-open byte range `[start, end)` in one owner's logical extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    /// First byte of the range, inclusive.
    pub start: usize,
    /// One past the last byte of the range, exclusive.
    pub end: usize,
}

impl ByteRange {
    /// Number of bytes in the range; zero when `end` does not exceed `start`.
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range holds no bytes.
    pub fn is_empty(self) -> bool {
        self.end <= self.start
    }
}

/// One placement row: either every listed device holds a full replica of `owner`, or `owner`'s logical
/// extent is split into exact disjoint device-owned ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementRow {
    /// Every listed device holds a full copy of the owner.
    Replicated {
        /// The owner that is replicated.
        owner: HostOwner,
        /// The devices holding a replica.
        residents: Vec<DeviceId>,
    },
    /// The owner's extent is split into disjoint device-owned ranges.
    Partitioned {
        /// The owner that is partitioned.
        owner: HostOwner,
        /// Each device with the range of the owner it holds.
        ranges: Vec<(DeviceId, ByteRange)>,
    },
}

/// Why a placement is not admitted.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    /// A partition boundary falls inside an encoding unit of the owner.
    #[error(
        "owner {owner:?} range [{start}, {end}) does not align to its {unit}-byte encoding unit"
    )]
    IllegalEncodingSplit {
        /// The owner whose range is misaligned.
        owner: OwnerId,
        /// Start of the misaligned range, in bytes.
        start: usize,
        /// End of the misaligned range, in bytes.
        end: usize,
        /// The owner's `encoding_unit_bytes`.
        unit: usize,
    },
    /// An owner appears in more than one placement row.
    #[error("owner {owner:?} appears in placement rows {indices:?}")]
    DuplicateOwner {
        /// The owner that is listed more than once.
        owner: OwnerId,
        /// Indices of the rows that name it.
        indices: Vec<usize>,
    },
    /// An owner the caller expects has no placement row.
    #[error("expected owner {owner:?} has no placement row")]
    MissingOwner {
        /// The owner with no row.
        owner: OwnerId,
    },
    /// A placement row names an owner outside the expected set.
    #[error("placement row {index} names owner {owner:?}, which is not in the expected set")]
    UnexpectedOwner {
        /// The unexpected owner.
        owner: OwnerId,
        /// Index of the row that names it.
        index: usize,
    },
    /// The placement puts bytes of `owner` on `device`, which hosts none of the stages that read it. The
    /// first reading stage is `stage`.
    #[error(
        "owner {owner:?} is placed on device {device:?}, which runs none of its stages (first reader: stage {stage})"
    )]
    OffStage {
        /// The owner placed on the foreign device.
        owner: OwnerId,
        /// The device that hosts none of the owner's reading stages.
        device: DeviceId,
        /// The first stage that reads the owner.
        stage: usize,
    },
}

/// Placement rows in which every expected owner appears in exactly one row and no row names an
/// owner outside the expected set. This is the set-level property only; row-level validity
/// (encoding alignment of a split) is the sharding pass's job.
/// Constructed solely by [`validate_placement_set`], so an owner set that is duplicated, missing
/// or unexpected cannot be represented once validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementSet {
    rows: Vec<PlacementRow>,
}

impl PlacementSet {
    /// The validated rows.
    pub fn rows(&self) -> &[PlacementRow] {
        &self.rows
    }
}

/// Authoritative [`HostOwner`] identity of one placement row, regardless of row shape.
fn row_owner(row: &PlacementRow) -> OwnerId {
    match row {
        PlacementRow::Replicated { owner, .. } | PlacementRow::Partitioned { owner, .. } => {
            owner.owner
        }
    }
}

/// Validate a placement SET against the caller's expected owner list (set-level, FR-002):
/// each expected owner appears in exactly one row, and every row names an expected owner.
///
/// On success returns a [`PlacementSet`]. `expected` is membership only - a repeated entry there
/// is redundant, not an error. Report order is unexpected (lowest row index), then duplicate
/// (first duplicated owner in row order), then missing (first gap in `expected` order).
///
pub fn validate_placement_set(
    rows: &[PlacementRow],
    expected: &[OwnerId],
) -> Result<PlacementSet, PlacementError> {
    let expected_set: HashSet<OwnerId> = expected.iter().copied().collect();
    let mut indices_by_owner: HashMap<OwnerId, Vec<usize>> = HashMap::new();
    let mut first_unexpected: Option<(usize, OwnerId)> = None;
    for (index, row) in rows.iter().enumerate() {
        let owner = row_owner(row);
        if !expected_set.contains(&owner) {
            if first_unexpected.is_none_or(|(seen, _)| index < seen) {
                first_unexpected = Some((index, owner));
            }
            continue;
        }
        indices_by_owner.entry(owner).or_default().push(index);
    }
    if let Some((index, owner)) = first_unexpected {
        return Err(PlacementError::UnexpectedOwner { owner, index });
    }
    for row in rows {
        let owner = row_owner(row);
        if let Some(indices) = indices_by_owner.get(&owner)
            && indices.len() > 1
        {
            return Err(PlacementError::DuplicateOwner {
                owner,
                indices: indices.clone(),
            });
        }
    }
    for &owner in expected {
        if !indices_by_owner.contains_key(&owner) {
            return Err(PlacementError::MissingOwner { owner });
        }
    }
    Ok(PlacementSet {
        rows: rows.to_vec(),
    })
}

/// [`validate_placement_set`], then the stage rule: every device an owner's placement reaches runs a
/// stage that reads that owner.
///
/// `readers[owner]` is the `(stage, device)` of every stage that reads `owner`; an owner no stage reads
/// has no entry, so any device it is placed on is off stage.
pub fn validate_stage_placement(
    rows: &[PlacementRow],
    expected: &[OwnerId],
    readers: &HashMap<OwnerId, Vec<(usize, DeviceId)>>,
) -> Result<PlacementSet, PlacementError> {
    let set = validate_placement_set(rows, expected)?;
    for row in set.rows() {
        let owner = row_owner(row);
        let stages = readers.get(&owner).map(Vec::as_slice).unwrap_or_default();
        let mut devices: Vec<DeviceId> = placement_device_bytes(row).into_keys().collect();
        devices.sort();
        for device in devices {
            if !stages
                .iter()
                .any(|&(_, stage_device)| stage_device == device)
            {
                return Err(PlacementError::OffStage {
                    owner,
                    device,
                    stage: stages.first().map_or(0, |&(stage, _)| stage),
                });
            }
        }
    }
    Ok(set)
}

/// Per-device byte contribution of one placement row (FR-002/FR-007): the only place device accounting
/// is derived from placement. A replicated owner is never duplicated on the host side (Card 360
/// constructs exactly one [`HostOwner`] value), but each device resident accounts the owner's full
/// `total_bytes` independently, since each holds its own device-side copy.
pub fn placement_device_bytes(row: &PlacementRow) -> HashMap<DeviceId, u64> {
    match row {
        PlacementRow::Replicated { owner, residents } => residents
            .iter()
            .map(|&device| (device, owner.total_bytes as u64))
            .collect(),
        PlacementRow::Partitioned { ranges, .. } => ranges
            .iter()
            .map(|&(device, range)| (device, range.len() as u64))
            .collect(),
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    impl PlacementSet {
        pub(crate) fn into_rows(self) -> Vec<PlacementRow> {
            self.rows
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
    pub(crate) enum PackedResidencyError {
        #[error(
            "device {device:?} accounts {bytes} bytes for owner {owner:?}, at or above the forbidden fully-\
             decoded f32 mirror size {forbidden_bytes} (logical shape [{out}, {k}])"
        )]
        DeviceDenseMirror {
            device: DeviceId,
            owner: OwnerId,
            bytes: u64,
            forbidden_bytes: u64,
            out: usize,
            k: usize,
        },
        #[error(
            "communication accounts {bytes} bytes for owner {owner:?}, at or above the forbidden fully-\
             decoded f32 mirror size {forbidden_bytes} (logical shape [{out}, {k}])"
        )]
        CommunicationDenseMirror {
            owner: OwnerId,
            bytes: u64,
            forbidden_bytes: u64,
            out: usize,
            k: usize,
        },
    }

    /// FR-011: no device or communication buffer may account bytes for a full decoded f32 mirror of a
    /// packed-quantized owner. `forbidden_bytes` is `4 * out * k`, the same size a decoded mirror of
    /// `poot_quant::PackedWeight`'s logical shape would occupy - Card 360 takes `out`/`k` from the
    /// caller's declared logical shape, never from a tensor name.
    pub(crate) fn reject_dense_quant_mirror(
        owner: OwnerId,
        logical_out: usize,
        logical_k: usize,
        device_bytes: &HashMap<DeviceId, u64>,
        communication_bytes: u64,
    ) -> Result<(), PackedResidencyError> {
        let forbidden_bytes = (logical_out as u64)
            .checked_mul(logical_k as u64)
            .and_then(|values| values.checked_mul(4))
            .unwrap_or(u64::MAX);
        for (&device, &bytes) in device_bytes {
            if bytes >= forbidden_bytes {
                return Err(PackedResidencyError::DeviceDenseMirror {
                    device,
                    owner,
                    bytes,
                    forbidden_bytes,
                    out: logical_out,
                    k: logical_k,
                });
            }
        }
        if communication_bytes >= forbidden_bytes {
            return Err(PackedResidencyError::CommunicationDenseMirror {
                owner,
                bytes: communication_bytes,
                forbidden_bytes,
                out: logical_out,
                k: logical_k,
            });
        }
        Ok(())
    }
}
