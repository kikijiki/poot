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
    pub owner: OwnerId,
    pub total_bytes: usize,
    /// Bytes per encoding-defined unit (e.g. one packed row); a partition boundary must land on a
    /// multiple of this, or on `total_bytes` itself. `1` means no encoding constraint - not every Card
    /// 359 owner is a packed payload.
    pub encoding_unit_bytes: usize,
}

/// A half-open byte range `[start, end)` in one owner's logical extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    pub start: usize,
    pub end: usize,
}

impl ByteRange {
    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.end <= self.start
    }
}

/// One placement row: either every listed device holds a full replica of `owner`, or `owner`'s logical
/// extent is split into exact disjoint device-owned ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementRow {
    Replicated {
        owner: HostOwner,
        residents: Vec<DeviceId>,
    },
    Partitioned {
        owner: HostOwner,
        ranges: Vec<(DeviceId, ByteRange)>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error("owner {owner:?} has zero total bytes")]
    EmptyOwner { owner: OwnerId },
    #[error("owner {owner:?} replicated placement has no residents")]
    NoResidents { owner: OwnerId },
    #[error("owner {owner:?} has an empty range [{start}, {end})")]
    EmptyRange {
        owner: OwnerId,
        start: usize,
        end: usize,
    },
    #[error("owner {owner:?} range [{start}, {end}) exceeds its logical extent {total_bytes}")]
    RangeOutOfBounds {
        owner: OwnerId,
        start: usize,
        end: usize,
        total_bytes: usize,
    },
    #[error("owner {owner:?} ranges [{a_start}, {a_end}) and [{b_start}, {b_end}) overlap")]
    RangeOverlap {
        owner: OwnerId,
        a_start: usize,
        a_end: usize,
        b_start: usize,
        b_end: usize,
    },
    #[error("owner {owner:?} coverage has a gap starting at byte {byte}")]
    CoverageGap { owner: OwnerId, byte: usize },
    #[error(
        "owner {owner:?} range [{start}, {end}) does not align to its {unit}-byte encoding unit"
    )]
    IllegalEncodingSplit {
        owner: OwnerId,
        start: usize,
        end: usize,
        unit: usize,
    },
    #[error("owner {owner:?} appears in placement rows {indices:?}")]
    DuplicateOwner { owner: OwnerId, indices: Vec<usize> },
    #[error("expected owner {owner:?} has no placement row")]
    MissingOwner { owner: OwnerId },
    #[error("placement row {index} names owner {owner:?}, which is not in the expected set")]
    UnexpectedOwner { owner: OwnerId, index: usize },
}

/// Validate exact disjoint coverage and encoding-unit alignment for one placement row (FR-003).
///
/// For [`PlacementRow::Partitioned`], ranges are sorted by start before adjacency is checked. Comparing
/// adjacent pairs in caller-supplied order would miss an overlap between two ranges that are not
/// adjacent in input order at three or more partitions (N=2 cannot exercise this; see the acceptance
/// table's "Partitioned (general)" row).
pub fn validate_placement(row: &PlacementRow) -> Result<(), PlacementError> {
    match row {
        PlacementRow::Replicated { owner, residents } => {
            if owner.total_bytes == 0 {
                return Err(PlacementError::EmptyOwner { owner: owner.owner });
            }
            if residents.is_empty() {
                return Err(PlacementError::NoResidents { owner: owner.owner });
            }
            Ok(())
        }
        PlacementRow::Partitioned { owner, ranges } => {
            if owner.total_bytes == 0 {
                return Err(PlacementError::EmptyOwner { owner: owner.owner });
            }
            let unit = owner.encoding_unit_bytes.max(1);
            for &(_, range) in ranges {
                if range.is_empty() {
                    return Err(PlacementError::EmptyRange {
                        owner: owner.owner,
                        start: range.start,
                        end: range.end,
                    });
                }
                if range.end > owner.total_bytes {
                    return Err(PlacementError::RangeOutOfBounds {
                        owner: owner.owner,
                        start: range.start,
                        end: range.end,
                        total_bytes: owner.total_bytes,
                    });
                }
                let end_aligned = range.end == owner.total_bytes || range.end % unit == 0;
                if range.start % unit != 0 || !end_aligned {
                    return Err(PlacementError::IllegalEncodingSplit {
                        owner: owner.owner,
                        start: range.start,
                        end: range.end,
                        unit,
                    });
                }
            }
            // Sort by start, not the caller's declaration order, so an overlap between ranges that are
            // non-adjacent in input order is still caught by an adjacent-pair scan.
            let mut sorted: Vec<ByteRange> = ranges.iter().map(|&(_, range)| range).collect();
            sorted.sort_by_key(|range| range.start);
            let mut cursor = 0usize;
            for range in &sorted {
                if range.start < cursor {
                    let overlapping = sorted
                        .iter()
                        .find(|other| other.start < range.start && other.end > range.start)
                        .copied()
                        .unwrap_or(ByteRange {
                            start: 0,
                            end: cursor,
                        });
                    return Err(PlacementError::RangeOverlap {
                        owner: owner.owner,
                        a_start: overlapping.start,
                        a_end: overlapping.end,
                        b_start: range.start,
                        b_end: range.end,
                    });
                }
                if range.start > cursor {
                    return Err(PlacementError::CoverageGap {
                        owner: owner.owner,
                        byte: cursor,
                    });
                }
                cursor = cursor.max(range.end);
            }
            if cursor < owner.total_bytes {
                return Err(PlacementError::CoverageGap {
                    owner: owner.owner,
                    byte: cursor,
                });
            }
            Ok(())
        }
    }
}

/// Placement rows in which every expected owner appears in exactly one row and no row names an
/// owner outside the expected set. This is the set-level property only; row-level validity
/// (empty owners, coverage gaps, encoding alignment) remains [`validate_placement`]'s job.
/// Constructed solely by [`validate_placement_set`], so an owner set that is duplicated, missing
/// or unexpected cannot be represented once validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementSet {
    rows: Vec<PlacementRow>,
}

impl PlacementSet {
    pub fn rows(&self) -> &[PlacementRow] {
        &self.rows
    }

    #[cfg(test)]
    pub(crate) fn into_rows(self) -> Vec<PlacementRow> {
        self.rows
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
/// Row-level validity is still checked separately by [`validate_placement`]; this function does
/// not subsume it.
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
#[cfg(test)]
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
