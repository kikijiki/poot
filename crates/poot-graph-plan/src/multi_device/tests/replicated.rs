//! Card 360 acceptance table row "Replicated" (N=2): one host owner, separately accounted device
//! residents, warm zero upload.
//!
//! Mutation that must fail: duplicate authoritative host bytes. See the card's "why this N": duplication
//! is a binary property of one owner vs. two residents, so N=2 already exercises it.

use crate::multi_device::communication::CommunicationPlan;
use crate::multi_device::placement::{HostOwner, OwnerId, PlacementRow, placement_device_bytes};
use crate::multi_device::replay::{ReplayContract, StaticBuffer, StaticBufferId, StaticBufferKind};
use crate::multi_device::topology::DeviceId;

use super::FakeReplayRuntime;

#[test]
fn multi_device_replicated_owner_accounts_each_resident_separately_and_replays_warm() {
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

    // Warm replay: two different device-resident copies (different ids, different fingerprints, so a
    // bug that only tracks device 0's buffer cannot hide behind symmetry) stay resident unchanged, so a
    // second run allocates and uploads nothing.
    let buffers = vec![
        StaticBuffer {
            id: StaticBufferId("weight:device0".into()),
            kind: StaticBufferKind::PackedWeight,
            bytes: 4096,
            fingerprint: 0xAAAA,
            per_replay_input: false,
        },
        StaticBuffer {
            id: StaticBufferId("weight:device1".into()),
            kind: StaticBufferKind::PackedWeight,
            bytes: 4096,
            fingerprint: 0xBBBB,
            per_replay_input: false,
        },
    ];
    let contract = ReplayContract {
        buffers,
        dispatch_count: 1,
        communication: CommunicationPlan::default(),
    };
    let mut runtime = FakeReplayRuntime::new();
    runtime.run(&contract);
    assert_eq!(runtime.counters.allocations, 2);
    assert_eq!(runtime.counters.uploads, 2);

    runtime.run(&contract);
    assert_eq!(
        runtime.counters.allocations, 2,
        "warm replay must not allocate"
    );
    assert_eq!(runtime.counters.uploads, 2, "warm replay must not reupload");
}
