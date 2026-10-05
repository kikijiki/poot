//! Set-level placement validation (duplicate / missing / unexpected owner across a row set).
//!
//! N=3 expected owners: at N<=1 a `collect` cannot drop or duplicate a row, and N=2 has only one
//! possible ordering, so "permuted order still valid" and "first-by-row-order duplicate" both need
//! three owners to be distinguishable from a set-keyed implementation.
//!
//! Mutations that must fail: drop the duplicate-owner check in `validate_placement_set`
//! (`multi_device_placement_set_rejects_a_duplicate_owner` must go red); drop the missing-owner
//! check (`multi_device_placement_set_rejects_a_missing_owner` must go red). Each row is a single
//! defect so an unexpected-owner pass cannot mask a duplicate (or vice versa).

use crate::multi_device::placement::{
    HostOwner, OwnerId, PlacementError, PlacementRow, validate_placement_set,
};
use crate::multi_device::topology::DeviceId;

const EXPECTED: [OwnerId; 3] = [OwnerId(1), OwnerId(2), OwnerId(3)];

fn replicated(owner: OwnerId) -> PlacementRow {
    PlacementRow::Replicated {
        owner: HostOwner {
            owner,
            total_bytes: 4096,
            encoding_unit_bytes: 1,
        },
        residents: vec![DeviceId(0)],
    }
}

fn owners_in(rows: &[PlacementRow]) -> Vec<OwnerId> {
    rows.iter()
        .map(|row| match row {
            PlacementRow::Replicated { owner, .. } | PlacementRow::Partitioned { owner, .. } => {
                owner.owner
            }
        })
        .collect()
}

#[test]
fn multi_device_placement_set_accepts_every_expected_owner_exactly_once() {
    let rows: Vec<_> = EXPECTED.iter().copied().map(replicated).collect();
    let set = validate_placement_set(&rows, &EXPECTED).expect("exact cover must validate");
    assert_eq!(set.rows(), rows.as_slice());
    assert_eq!(set.into_rows(), rows);
}

#[test]
fn multi_device_placement_set_accepts_a_permuted_row_order() {
    // Non-reversal permutation of three owners: a validator that required expected order (or that
    // sorted rows into a map keyed only by owner id before comparing sequences) would reject it.
    let rows = vec![
        replicated(OwnerId(3)),
        replicated(OwnerId(1)),
        replicated(OwnerId(2)),
    ];
    let set = validate_placement_set(&rows, &EXPECTED).expect("permuted cover must validate");
    assert_eq!(
        owners_in(set.rows()),
        vec![OwnerId(3), OwnerId(1), OwnerId(2)],
        "validated set must retain caller row order, not expected order"
    );
}

#[test]
fn multi_device_placement_set_rejects_a_duplicate_owner() {
    // Single defect: every expected owner appears, owner 1 appears twice. A check that only tested
    // membership (or only missing/unexpected) would accept this.
    let rows = vec![
        replicated(OwnerId(1)),
        replicated(OwnerId(2)),
        replicated(OwnerId(3)),
        replicated(OwnerId(1)),
    ];
    assert_eq!(
        validate_placement_set(&rows, &EXPECTED),
        Err(PlacementError::DuplicateOwner {
            owner: OwnerId(1),
            indices: vec![0, 3],
        })
    );

    // Duplicate of a later owner, first occurrence not at row 0: indices must be the real rows.
    let rows = vec![
        replicated(OwnerId(1)),
        replicated(OwnerId(2)),
        replicated(OwnerId(3)),
        replicated(OwnerId(2)),
    ];
    assert_eq!(
        validate_placement_set(&rows, &EXPECTED),
        Err(PlacementError::DuplicateOwner {
            owner: OwnerId(2),
            indices: vec![1, 3],
        })
    );
}

#[test]
fn multi_device_placement_set_rejects_a_missing_owner() {
    // Single defect: owners 1 and 2 present once each; expected owner 3 has no row. Duplicates
    // and unexpected owners are absent, so only the missing check can reject.
    let rows = vec![replicated(OwnerId(1)), replicated(OwnerId(2))];
    assert_eq!(
        validate_placement_set(&rows, &EXPECTED),
        Err(PlacementError::MissingOwner { owner: OwnerId(3) })
    );

    // Gap in the middle of expected order: report the first missing expected owner, not row order.
    let rows = vec![replicated(OwnerId(1)), replicated(OwnerId(3))];
    assert_eq!(
        validate_placement_set(&rows, &EXPECTED),
        Err(PlacementError::MissingOwner { owner: OwnerId(2) })
    );
}

#[test]
fn multi_device_placement_set_rejects_an_unexpected_owner() {
    // Single defect relative to a complete cover: all three expected owners present once, plus
    // row 3 names owner 9 which is not expected.
    let rows = vec![
        replicated(OwnerId(1)),
        replicated(OwnerId(2)),
        replicated(OwnerId(3)),
        replicated(OwnerId(9)),
    ];
    assert_eq!(
        validate_placement_set(&rows, &EXPECTED),
        Err(PlacementError::UnexpectedOwner {
            owner: OwnerId(9),
            index: 3,
        })
    );

    // Unexpected owner alone (empty expected set with a non-empty row list).
    let rows = vec![replicated(OwnerId(1))];
    assert_eq!(
        validate_placement_set(&rows, &[]),
        Err(PlacementError::UnexpectedOwner {
            owner: OwnerId(1),
            index: 0,
        })
    );
}

#[test]
fn multi_device_placement_set_empty_rows_match_empty_expectations() {
    let set =
        validate_placement_set(&[], &[]).expect("empty expected with no rows is an exact cover");
    assert!(set.rows().is_empty());
    assert!(set.into_rows().is_empty());

    // Non-empty expected with no rows reports the first missing owner, not Ok.
    assert_eq!(
        validate_placement_set(&[], &EXPECTED),
        Err(PlacementError::MissingOwner { owner: OwnerId(1) })
    );
}
