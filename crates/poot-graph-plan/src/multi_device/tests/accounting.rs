//! Card 360 acceptance table row "Per-device capacity" (N=3, deficient device in the middle position).
//!
//! Mutation that must fail: replace the weakest-device gate with aggregate capacity. N=2 cannot rule out
//! an implementation that only checks the first or last device; a middle-position failure rules that out.

use std::collections::HashMap;

use crate::multi_device::accounting::{AccountingError, ByteCategories, admit_per_device_capacity};
use crate::multi_device::tests::fixtures::device;
use crate::multi_device::topology::DeviceId;

#[test]
fn multi_device_accounting_is_per_device_checked() {
    let devices = vec![device(0, 1_000_000), device(1, 100), device(2, 1_000_000)];
    let mut usage = HashMap::new();
    usage.insert(
        DeviceId(0),
        ByteCategories {
            source_or_carrier: 900_000,
            ..Default::default()
        },
    );
    // Device 1, in the middle position, is one byte over its own 100-byte usable capacity.
    usage.insert(
        DeviceId(1),
        ByteCategories {
            source_or_carrier: 101,
            ..Default::default()
        },
    );
    usage.insert(
        DeviceId(2),
        ByteCategories {
            source_or_carrier: 900_000,
            ..Default::default()
        },
    );

    // Aggregate usable (2,000,100) comfortably covers aggregate used (1,800,101); only the per-device
    // check distinguishes this from an admissible plan.
    assert_eq!(
        admit_per_device_capacity(&devices, &usage),
        Err(AccountingError::CapacityExceeded {
            device: DeviceId(1),
            required: 101,
            usable: 100,
        })
    );
}
