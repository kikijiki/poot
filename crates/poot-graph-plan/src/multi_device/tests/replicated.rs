//! Card 360 acceptance table row "Replicated" (N=2): one host owner, separately accounted device
//! residents.
//!
//! Mutation that must fail: duplicate authoritative host bytes. See the card's "why this N": duplication
//! is a binary property of one owner vs. two residents, so N=2 already exercises it.

use crate::multi_device::placement::{HostOwner, OwnerId, PlacementRow, placement_device_bytes};
use crate::multi_device::topology::DeviceId;

#[test]
fn multi_device_replicated_owner_accounts_each_resident_separately() {
    let owner = HostOwner {
        owner: OwnerId(1),
        total_bytes: 4096,
        encoding_unit_bytes: 1,
    };
    let dev0 = DeviceId(0);
    let dev1 = DeviceId(1);
    let row = PlacementRow::Replicated {
        owner,
        residents: vec![dev0, dev1],
    };

    // One authoritative owner: both residents name the same OwnerId; Card 360 never constructs a
    // second owner value for a replicated placement.
    let PlacementRow::Replicated {
        owner: row_owner,
        residents,
    } = &row
    else {
        panic!("expected a replicated row");
    };
    assert_eq!(row_owner.owner, OwnerId(1));
    assert_eq!(residents, &vec![dev0, dev1]);

    // Each resident is accounted separately at the owner's full logical size, not halved between the
    // two devices, not merged into a single device's tally.
    let bytes = placement_device_bytes(&row);
    assert_eq!(bytes.get(&dev0), Some(&4096));
    assert_eq!(bytes.get(&dev1), Some(&4096));
}
