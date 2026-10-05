//! Card 360 acceptance table row "Replay-contract stability (fake runtime)" (N=2): warm replay has zero
//! constant upload, carrier rebuild, or allocation.
//!
//! Mutation that must fail: recreate one unchanged scale or communication buffer. Stability is per-
//! buffer, not an ordering claim, so N=2 suffices - but the two buffers below use different ids, kinds,
//! byte sizes, and fingerprints, so a bug that only tracks one device-index symmetrically cannot hide.

use crate::multi_device::communication::CommunicationPlan;
use crate::multi_device::replay::{ReplayContract, StaticBuffer, StaticBufferId, StaticBufferKind};

use super::FakeReplayRuntime;

#[test]
fn multi_device_warm_replay_is_stable() {
    let buffers = vec![
        StaticBuffer {
            id: StaticBufferId("scale:device0".into()),
            kind: StaticBufferKind::Scale,
            bytes: 128,
            fingerprint: 0x1111,
            per_replay_input: false,
        },
        StaticBuffer {
            id: StaticBufferId("weight:device1".into()),
            kind: StaticBufferKind::PackedWeight,
            bytes: 4096,
            fingerprint: 0x2222,
            per_replay_input: false,
        },
        StaticBuffer {
            id: StaticBufferId("comm:allreduce-0".into()),
            kind: StaticBufferKind::Communication,
            bytes: 64,
            fingerprint: 0x3333,
            per_replay_input: false,
        },
    ];
    let contract = ReplayContract {
        buffers,
        dispatch_count: 2,
        communication: CommunicationPlan::default(),
    };
    let mut runtime = FakeReplayRuntime::new();

    runtime.run(&contract);
    assert_eq!(runtime.counters.allocations, 3);
    assert_eq!(runtime.counters.uploads, 3);
    assert_eq!(runtime.counters.dispatches, 2);

    runtime.run(&contract);
    assert_eq!(
        runtime.counters.allocations, 3,
        "warm replay must not allocate"
    );
    assert_eq!(
        runtime.counters.uploads, 3,
        "warm replay must not reupload an unchanged scale or communication buffer"
    );
    assert_eq!(runtime.counters.carrier_rebuilds, 0);
    assert_eq!(
        runtime.counters.dispatches, 4,
        "dispatches still occur warm - only allocation/upload/rebuild must go to zero"
    );
}
