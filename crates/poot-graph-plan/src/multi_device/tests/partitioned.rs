//! Card 360 acceptance table rows "Partitioned (two-way)" (N=2) and "Partitioned (general)" (N=3).
//!
//! Two-way mutation that must fail: add a one-byte overlap or gap.
//! General mutation that must fail: add a one-byte overlap between two non-adjacent ranges. A coverage
//! checker that only compares adjacent pairs in caller-declared order (rather than sorting by start
//! first) would pass at N=2 - where "adjacent in input order" and "adjacent when sorted" always
//! coincide - and only diverges once a third range lets an overlapping pair sit apart in input order.

use crate::multi_device::placement::{
    ByteRange, HostOwner, OwnerId, PlacementError, PlacementRow, validate_placement,
};
use crate::multi_device::topology::DeviceId;

fn owner(total_bytes: usize) -> HostOwner {
    HostOwner {
        owner: OwnerId(7),
        total_bytes,
        encoding_unit_bytes: 1,
    }
}

#[test]
fn multi_device_placement_covers_source_exactly_two_way() {
    let exact = PlacementRow::Partitioned {
        owner: owner(100),
        ranges: vec![
            (DeviceId(0), ByteRange { start: 0, end: 60 }),
            (
                DeviceId(1),
                ByteRange {
                    start: 60,
                    end: 100,
                },
            ),
        ],
    };
    assert!(validate_placement(&exact).is_ok());

    let one_byte_gap = PlacementRow::Partitioned {
        owner: owner(100),
        ranges: vec![
            (DeviceId(0), ByteRange { start: 0, end: 59 }),
            (
                DeviceId(1),
                ByteRange {
                    start: 60,
                    end: 100,
                },
            ),
        ],
    };
    assert_eq!(
        validate_placement(&one_byte_gap),
        Err(PlacementError::CoverageGap {
            owner: OwnerId(7),
            byte: 59
        })
    );

    let one_byte_overlap = PlacementRow::Partitioned {
        owner: owner(100),
        ranges: vec![
            (DeviceId(0), ByteRange { start: 0, end: 61 }),
            (
                DeviceId(1),
                ByteRange {
                    start: 60,
                    end: 100,
                },
            ),
        ],
    };
    assert!(matches!(
        validate_placement(&one_byte_overlap),
        Err(PlacementError::RangeOverlap { .. })
    ));
}

#[test]
fn multi_device_placement_covers_source_exactly_general() {
    // Declaration order is [X, Y, Z]. X and Z overlap by exactly one byte (X=[0,100), Z=[99,200)); Y
    // sits between them in the logical extent and is declared between them, so the overlapping pair is
    // NOT adjacent in input order. Sorting by start first (X, Z, Y by start: 0, 99, 200) puts the
    // overlap where an adjacent-pair scan finds it; scanning input-order-adjacent pairs (X,Y) then (Y,Z)
    // never compares X against Z at all.
    let x = (DeviceId(0), ByteRange { start: 0, end: 100 });
    let y = (
        DeviceId(1),
        ByteRange {
            start: 200,
            end: 300,
        },
    );
    let z = (
        DeviceId(2),
        ByteRange {
            start: 99,
            end: 200,
        },
    );
    let non_adjacent_overlap = PlacementRow::Partitioned {
        owner: owner(300),
        ranges: vec![x, y, z],
    };
    assert!(matches!(
        validate_placement(&non_adjacent_overlap),
        Err(PlacementError::RangeOverlap { .. })
    ));

    // Sanity: the equivalent exact partition (no overlap) at N=3 is accepted.
    let exact = PlacementRow::Partitioned {
        owner: owner(300),
        ranges: vec![
            (DeviceId(0), ByteRange { start: 0, end: 100 }),
            (
                DeviceId(1),
                ByteRange {
                    start: 200,
                    end: 300,
                },
            ),
            (
                DeviceId(2),
                ByteRange {
                    start: 100,
                    end: 200,
                },
            ),
        ],
    };
    assert!(validate_placement(&exact).is_ok());
}
